import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import pyttsx3
import time
from collections import deque, defaultdict


# ------------------ Configuration / Hyperparameters ------------------
SMOOTHING_WINDOW = 5           # number of frames to smooth joint coords / angles
MIN_VISIBILITY = 0.45          # minimum average visibility to accept a side
POSTURE_HOLD_SEC = 0.4         # how long the good posture must hold before counting reps
REP_COOLDOWN = 0.9             # default minimum seconds between counted reps
FAST_REP_COOLDOWN = 0.30       # faster alternating exercises
VOICE_BACKGROUND = True        # speak in background thread to avoid blocking

# FIX (rep counter / form score bugs):
#   How many *consecutive* bad-posture frames are tolerated before we reset
#   the "posture is holding" timer. Previously a single noisy frame reset
#   posture_ok_since to None, which combined with an unrealistically tight
#   back_angle threshold (>80 on a 150-180 range, i.e. >=174 degrees) meant
#   reps almost never counted. See BICEP['back_ok'] below for the other half
#   of this fix.
POSTURE_BAD_FRAME_TOLERANCE = 3

# FIX (posture_score / correct_posture disagreement — SIH review):
#   Several exercises (Lunges, Shoulder Press, Push-ups, Glute Bridge) used
#   to compute `correct_posture` from a raw angle threshold and
#   `posture_score` from interp_clip() on the SAME angle, but with
#   different cutoffs. That meant the boolean could say "good posture"
#   while the numeric score was near 0 (e.g. torso_angle just barely above
#   140 -> correct_posture True, but interp_clip(140,(140,180)) -> score 0).
#   This single shared constant is now the ONE place that defines "how good
#   does the numeric score have to be for we call it correct posture". All
#   four exercises below derive their boolean from the score using this
#   value instead of re-checking the raw angle a second time. Raised to 70
#   (from an initial 50) per project requirement: posture must be judged
#   strictly, not just "better than half-bad".
POSTURE_SCORE_PASS_THRESHOLD = 70

# Angle thresholds / mapping (kept together so tuning is easy)
BICEP = {
    # FIX: widened from (150, 180). 150-180 forced the user to be within
    # ~6 degrees of ramrod-straight to ever register "correct posture",
    # which is unrealistic for a bicep curl (some natural sway is normal).
    "back_angle_range": (140, 180),
    # FIX: lowered from 80 -> 55. On the new (140,180) range this now
    # requires back_angle >= ~162 degrees instead of >=174 degrees.
    # The old value made correct_posture essentially unreachable, which
    # cascaded into the rep counter (reps only count while posture is
    # "ready") and the form score (score was computed directly from this
    # same interpolation, so it looked volatile/harsh for the same reason).
    "back_ok": 55,
    "elbow_range": (160, 40),         # (straight, flexed)
    "progress_down_thresh": 15,
    "progress_up_thresh": 90,
    # FIX (elbow flare / "hand behind" never detected): correct_posture
    # previously only looked at back straightness. The feedback text
    # promises "elbow close to your torso" but nothing in the code
    # measured that, so flaring the elbow out / swinging it backward
    # still scored as "Good Form". This threshold caps how far the elbow
    # may drift from the shoulder — combining sideways (x) AND depth (z)
    # drift — as a fraction of torso length. Raised slightly from the
    # x-only version (0.35 -> 0.5) because adding the noisier z axis
    # raises the baseline reading even for genuinely good form; tune this
    # by eye if it's still too strict/loose for your camera setup.
    "elbow_offset_max": 0.5,
}

SQUAT = {
    "back_angle_range": (140, 180),
    "back_ok": 78,
    "knee_range": (170, 60),
    "progress_down_thresh": 85,   # progress (0=standing, 100=deep squat) crosses this -> bottom reached
    "progress_up_thresh": 20,     # progress crosses back below this -> returned to standing = 1 rep
    # FIX (knee tracking never checked): correct_posture previously only
    # looked at back angle. "Knees caving in" / "knee traveling far past
    # the toes" -- the most common squat form fault -- was never measured
    # even though it's a standard coaching cue. This caps how far the
    # knee may drift horizontally from the ankle, as a fraction of shin
    # length (ankle-to-knee distance), so it scales with camera distance.
    "knee_track_max": 0.55,
}

PLANK = {
    "ref_angle_range": (140, 180),
    "back_ok": 85,
    "hold_seconds": 20
}

PUSHUP = {
    "down_threshold": 90,
    "up_threshold": 160,
    "min_visibility": 0.7,
    "hip_sag_tolerance": 0.08
}

LUNGE = {
    "down_threshold": 100,
    "up_threshold": 160,
    "min_visibility": 0.7,
    "balance_tolerance": 0.10
}

SHOULDER_PRESS = {
    "up_threshold": 160,
    "down_threshold": 90,
    "min_visibility": 0.7
}

GLUTE_BRIDGE = {
    "up_threshold": 160,
    "down_threshold": 100,
    "min_visibility": 0.7
}

MOUNTAIN_CLIMBER = {
    "knee_threshold": 0.95,
    "min_visibility": 0.65,
    "body_line_min": 150,
    "hip_sag_tolerance": 0.10,
    "hip_pike_tolerance": 0.12
}

JUMPING_JACKS = {
    "arm_up_threshold": 0.75,
    "leg_open_threshold": 1.25,
    "min_visibility": 0.65
}

HIGH_KNEES = {
    "knee_up_threshold": 0.50,
    "min_visibility": 0.65,
    "torso_min": 155
}

SIDE_LUNGE = {
    # Moderate side-lunge range: do not force a very deep bend.
    # Around 120-130° at the working knee is enough for rep detection.
    "down_threshold": 125,
    "up_threshold": 155,
    "min_visibility": 0.7,
    # Small lateral shift is enough to distinguish a side lunge from a squat.
    "hip_shift_threshold": 0.08
}

SIDE_LEG_RAISE = {
    "up_threshold": 0.18,
    "down_threshold": 0.06,
    "min_visibility": 0.7,
    "max_knee_bend": 150
}

WALL_SIT = {
    "knee_angle_target": 90,
    "knee_angle_tolerance": 20,
    "min_visibility": 0.7,
    "hold_seconds": 20
}

STANDING_KNEE_ELBOW = {
    "knee_up_threshold": 0.45,
    "min_visibility": 0.65,
    "torso_min": 150,
    "max_lean": 0.18
}

ARM_CIRCLES = {
    "circle_radius": 0.18,
    "min_visibility": 0.65,
    "cooldown": 0.5,
    "rotation_target": 300,
    "min_arm_extension": 0.22,
    "min_delta": 0.5
}




# ------------------ Speech Engine (non-blocking wrapper) ------------------
# WEB_MODE: set to True when this module is imported by the FastAPI backend.
# In web mode the TTS engine is never initialised so importing the module
# from a headless server does not crash.
WEB_MODE = False

_engine = None
_speech_lock = threading.Lock()

def _get_engine():
    """Lazily initialise pyttsx3 only when running in standalone desktop mode."""
    global _engine
    if _engine is None:
        try:
            import pyttsx3 as _pyttsx3
            _engine = _pyttsx3.init()
            _engine.setProperty('rate', 170)
        except Exception:
            _engine = None
    return _engine

def speak(text):
    """Speak text aloud.  Silently skipped in WEB_MODE or if TTS is unavailable."""
    if not text or WEB_MODE:
        return
    engine = _get_engine()
    if engine is None:
        return
    if VOICE_BACKGROUND:
        def _bg():
            with _speech_lock:
                engine.say(text)
                engine.runAndWait()
        threading.Thread(target=_bg, daemon=True).start()
    else:
        with _speech_lock:
            engine.say(text)
            engine.runAndWait()

# ------------------ Utilities: smoothing and helpers ------------------
class LandmarkSmoother:
    """Keep recent coordinates for each landmark and provide smoothed values."""
    def __init__(self, window=SMOOTHING_WINDOW):
        self.window = window
        self.buffers = defaultdict(lambda: deque(maxlen=self.window))

    def add(self, idx, x, y, visibility):
        self.buffers[idx].append((x, y, visibility))

    def smoothed(self, idx):
        buf = self.buffers[idx]
        if not buf:
            return None
        arr = np.array(buf)
        mean_x, mean_y, mean_vis = np.mean(arr[:,0]), np.mean(arr[:,1]), np.mean(arr[:,2])
        return mean_x, mean_y, mean_vis

    def clear(self):
        self.buffers.clear()


def calculate_angle(a, b, c):
    """Return angle ABC (in degrees) robustly.
    a, b, c are 2-element iterables (x,y)
    """
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.degrees(np.abs(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def interp_clip(value, src_range, dst_range=(0,100)):
    """
    FIX (Squats not progressing / Bicep Curl janky rep detection):
    np.interp() REQUIRES its xp (here: src_range) to be increasing. Several
    exercises intentionally pass a *decreasing* src_range to express
    "high angle = start, low angle = end" (e.g. SQUAT['knee_range'] =
    (170, 60), BICEP['elbow_range'] = (160, 40)). numpy does not raise an
    error for a decreasing xp -- it silently returns near-constant garbage
    (pinned at one end of dst_range for almost the entire input domain,
    only snapping to the other end in a razor-thin band at the very
    extreme). That meant `progress` for Squats stayed near 100 for nearly
    the whole squat and only dropped to 0 at the very deepest point, so
    the down/up rep state machine (which expects a smooth ramp) rarely
    saw a real crossing -- squats looked "stuck"/unresponsive. Bicep Curl
    had the identical issue, just less visible because reps occasionally
    got counted via a lucky crossing right at full contraction.

    Fix: detect a decreasing src_range and flip both src_range and
    dst_range together before calling np.interp, which preserves the
    exact same intended mapping direction while keeping xp increasing
    (which np.interp actually requires to work correctly).
    """
    src_lo, src_hi = src_range
    dst_lo, dst_hi = dst_range
    if src_lo <= src_hi:
        out = np.interp(value, [src_lo, src_hi], [dst_lo, dst_hi])
    else:
        out = np.interp(value, [src_hi, src_lo], [dst_hi, dst_lo])
    lo, hi = min(dst_lo, dst_hi), max(dst_lo, dst_hi)
    return float(np.clip(out, lo, hi))

# ------------------ Side selection improved: choose the most visible side or both

def side_and_visibility(landmarks, mp_pose, smoother=None):
    """
    FIX (pose-dropout / 'No pose detected' flicker, side flip-flopping):
    Previously this read RAW per-frame landmarks.visibility. A single noisy
    frame (motion blur, brief occlusion, leaning toward the camera) could
    drop visibility below MIN_VISIBILITY and immediately flip the chosen
    side or report "not visible", even though LandmarkSmoother already
    existed and was used everywhere else in the pipeline. Routing this
    through the smoother (when available) makes side selection and the
    visibility gate consistent with the rest of the tracking logic and
    much less jumpy.
    """
    left_joints = [mp_pose.PoseLandmark.LEFT_SHOULDER,
                   mp_pose.PoseLandmark.LEFT_ELBOW,
                   mp_pose.PoseLandmark.LEFT_WRIST,
                   mp_pose.PoseLandmark.LEFT_HIP,
                   mp_pose.PoseLandmark.LEFT_KNEE]
    right_joints = [mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                    mp_pose.PoseLandmark.RIGHT_HIP,
                    mp_pose.PoseLandmark.RIGHT_KNEE]

    def vis_of(idx):
        if smoother is not None:
            sm = smoother.smoothed(idx)
            if sm is not None:
                return sm[2]
        return landmarks[idx].visibility

    left_vis = np.mean([vis_of(j.value) for j in left_joints])
    right_vis = np.mean([vis_of(j.value) for j in right_joints])

    # if both visible, return BOTH so logic can choose best per-joint
    if left_vis >= MIN_VISIBILITY and right_vis >= MIN_VISIBILITY:
        return "BOTH", max(left_vis, right_vis)
    side = "LEFT" if left_vis >= right_vis else "RIGHT"
    return side, max(left_vis, right_vis)

# ------------------ Exercise logic (cleaner, more robust)

def exercise_logic(exercise_name, landmarks, mp_pose, smoother,
                   state, now, min_visibility=MIN_VISIBILITY):
    """Robust exercise counter.

    Important design change:
      * repetition detection is based on movement, not on ``correct_posture``.
      * posture is reported separately and therefore a small form error cannot
        freeze the repetition counter.
      * all movement signals are smoothed and use hysteresis (different start
        and end thresholds), which prevents jitter/double-counting.
    """
    def lm(name, side):
        idx = getattr(mp_pose.PoseLandmark, f"{side}_{name}").value
        sm = smoother.smoothed(idx)
        if sm is not None:
            return np.array(sm[:2], dtype=float), float(sm[2])
        x = landmarks[idx].x
        y = landmarks[idx].y
        return np.array([x, y], dtype=float), float(landmarks[idx].visibility)

    def angle(a, b, c):
        return calculate_angle(a, b, c)

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    def avg(a, b):
        return (a + b) / 2.0

    def point_line_distance(point, line_a, line_b):
        """Perpendicular distance from point to the finite line segment."""
        ab = line_b - line_a
        denom = float(np.dot(ab, ab))
        if denom < 1e-8:
            return dist(point, line_a)
        t = float(np.dot(point - line_a, ab) / denom)
        t = np.clip(t, 0.0, 1.0)
        projection = line_a + t * ab
        return dist(point, projection)

    def bilateral_mean(left_value, right_value):
        return float((left_value + right_value) / 2.0)

    def count_if(condition, stage_name="ready"):
        nonlocal stage, count, last_rep_time
        if condition and stage != stage_name:
            stage = stage_name
            if now - last_rep_time >= REP_COOLDOWN:
                count += 1
                last_rep_time = now
                return True
        return False

    stage = state.get('stage')
    count = state.get('count', 0)
    last_rep_time = state.get('last_rep_time', 0.0)
    hold_start = state.get('hold_start', 0.0)
    posture_ok_since = state.get('posture_ok_since')

    # Smoothed coordinates for both sides.
    LS, lsv = lm('SHOULDER', 'LEFT')
    LE, lev = lm('ELBOW', 'LEFT')
    LW, lwv = lm('WRIST', 'LEFT')
    LH, lhv = lm('HIP', 'LEFT')
    LK, lkv = lm('KNEE', 'LEFT')
    LA, lav = lm('ANKLE', 'LEFT')
    RS, rsv = lm('SHOULDER', 'RIGHT')
    RE, rev = lm('ELBOW', 'RIGHT')
    RW, rwv = lm('WRIST', 'RIGHT')
    RH, rhv = lm('HIP', 'RIGHT')
    RK, rkv = lm('KNEE', 'RIGHT')
    RA, rav = lm('ANKLE', 'RIGHT')

    torso = max(dist(avg(LS, RS), avg(LH, RH)), 0.08)
    shoulder_width = max(dist(LS, RS), 0.08)
    hip_width = max(dist(LH, RH), 0.05)

    # Visibility should not be so strict that normal motion freezes counting.
    arm_vis = min(lsv, lwv, rsv, rwv)
    leg_vis = min(lhv, lkv, lav, rhv, rkv, rav)
    all_vis = min(arm_vis, leg_vis)

    correct_posture = True
    posture_score = 100.0
    progress = 0.0
    movement_detected = False

    def smooth_value(key, value, size=SMOOTHING_WINDOW):
        buf = state.setdefault(key, deque(maxlen=size))
        buf.append(float(value))
        return float(np.mean(buf))

    def set_posture(score):
        nonlocal correct_posture, posture_score
        posture_score = float(np.clip(score, 0, 100))
        correct_posture = posture_score >= POSTURE_SCORE_PASS_THRESHOLD

    # ------------------ Bicep Curl ------------------
    if exercise_name == 'Bicep Curl':
        side = 'LEFT' if lwv >= rwv else 'RIGHT'
        S, sv = (LS, lsv) if side == 'LEFT' else (RS, rsv)
        E, ev = (LE, lev) if side == 'LEFT' else (RE, rev)
        W, wv = (LW, lwv) if side == 'LEFT' else (RW, rwv)
        H, hv = (LH, lhv) if side == 'LEFT' else (RH, rhv)
        K, kv = (LK, lkv) if side == 'LEFT' else (RK, rkv)
        elbow_angle = smooth_value('bicep_angle', angle(S, E, W), 5)
        progress = interp_clip(elbow_angle, BICEP['elbow_range'])

        # Bicep posture must be judged from the actual curl position, not
        # from the raw distance between elbow and shoulder.  That old check
        # incorrectly marked a person standing normally as bad because the
        # elbow is naturally far below the shoulder, and it also became
        # unstable while curling.
        back_angle = smooth_value('bicep_back', angle(S, H, K), 7)

        # -------- Proper torso-bending detection --------
        # Measure the torso itself against vertical. Do not use the knee as
        # the reference because knee movement can hide a bent upper body.
        shoulder_mid = avg(LS, RS)
        hip_mid = avg(LH, RH)
        torso_dx = shoulder_mid[0] - hip_mid[0]
        torso_dy = shoulder_mid[1] - hip_mid[1]
        torso_bend = np.degrees(np.arctan2(abs(torso_dx), abs(torso_dy) + 1e-6))
        torso_bend = smooth_value('bicep_torso_bend', torso_bend, 7)

        # Natural sway is allowed. Noticeable bending progressively lowers
        # the score, and >=15 degrees is treated as incorrect form.
        bend_score = interp_clip(torso_bend, (0, 20), (100, 0))

        elbow_horizontal_offset = abs(E[0] - S[0]) / shoulder_width
        elbow_score = interp_clip(elbow_horizontal_offset, (0.08, 0.55), (100, 0))

        if side == 'LEFT':
            shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value
            wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value
        else:
            shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
            wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value

        shoulder_z = smooth_value('bicep_shoulder_z', landmarks[shoulder_idx].z, 7)
        elbow_z = smooth_value('bicep_elbow_z', landmarks[elbow_idx].z, 7)
        wrist_z = smooth_value('bicep_wrist_z', landmarks[wrist_idx].z, 7)
        elbow_back = elbow_z - shoulder_z
        wrist_back = wrist_z - shoulder_z

        elbow_depth_score = interp_clip(elbow_back, (0.02, 0.18), (100, 0))
        wrist_depth_score = interp_clip(wrist_back, (0.02, 0.22), (100, 0))
        depth_score = elbow_depth_score * 0.45 + wrist_depth_score * 0.55

        posture_score = bend_score * 0.55 + elbow_score * 0.20 + depth_score * 0.25

        # Hard faults: a clearly bent torso or clearly backward hand/elbow
        # must never be classified as good posture.
        if torso_bend >= 15 or wrist_back > 0.16 or elbow_back > 0.13:
            posture_score = min(posture_score, 55)

        set_posture(posture_score)

        # Bicep reps need a genuine curl, not just one noisy elbow-angle
        # frame.  A false positive used to happen when MediaPipe briefly
        # produced an unrealistically small elbow angle while the user was
        # standing still: that set stage='down', and the normal straight-arm
        # angle on the next frames immediately counted a rep.
        #
        # Require BOTH: (1) a sustained deep flexion and (2) the wrist to move
        # above the elbow.  The release back to extension also uses hysteresis.
        curl_depth = elbow_angle < 65
        wrist_above_elbow = W[1] < E[1] - 0.025
        genuine_curl = curl_depth and wrist_above_elbow and wv >= 0.50 and ev >= 0.50

        low_frames = state.get('bicep_low_frames', 0)
        if genuine_curl:
            low_frames += 1
        else:
            low_frames = max(0, low_frames - 1)
        state['bicep_low_frames'] = low_frames

        # Require several consecutive frames at the bottom so a single
        # landmark glitch can never arm the rep counter.
        if low_frames >= 4:
            stage = 'down'

        # Require a clearly extended arm before accepting the return as a rep.
        # This gives the counter real hysteresis: deep curl -> extension,
        # rather than noisy angle -> noisy angle.
        elif elbow_angle > 150 and stage == 'down' and now - last_rep_time >= REP_COOLDOWN:
            count += 1
            last_rep_time = now
            stage = 'up'
            state['bicep_low_frames'] = 0
            speak(f"Repetition {count}")

    # ------------------ Squat ------------------
    elif exercise_name == 'Squats':
        left_knee = smooth_value('squat_lk', angle(LH, LK, LA), 5)
        right_knee = smooth_value('squat_rk', angle(RH, RK, RA), 5)
        # Use the mean for progress so one temporarily noisy knee cannot
        # control the entire rep. Keep the minimum only as a safety signal.
        knee_angle = bilateral_mean(left_knee, right_knee)
        worst_knee = min(left_knee, right_knee)

        progress = interp_clip(knee_angle, (175, 115))

        torso_angle = smooth_value(
            'squat_torso',
            angle(avg(LS, RS), avg(LH, RH), avg(LK, RK)),
            7
        )

        # Knee-over-ankle tracking. This is deliberately normalized by shin
        # length, so it remains useful at different camera distances.
        left_track = abs(LK[0] - LA[0]) / max(dist(LK, LA), 0.08)
        right_track = abs(RK[0] - RA[0]) / max(dist(RK, RA), 0.08)
        knee_track = bilateral_mean(left_track, right_track)
        knee_track_score = interp_clip(knee_track, (0.18, SQUAT['knee_track_max']), (100, 0))

        # Do not let a single bad knee destroy a good reading, but clearly
        # asymmetric/collapsing tracking is still penalized.
        knee_asymmetry = abs(left_knee - right_knee)
        symmetry_score = interp_clip(knee_asymmetry, (8, 28), (100, 0))
        back_score = interp_clip(torso_angle, (135, 170), (0, 100))

        posture_score = back_score * 0.50 + knee_track_score * 0.30 + symmetry_score * 0.20
        if torso_angle < 130 or knee_track > 0.70:
            posture_score = min(posture_score, 55)
        set_posture(posture_score)

        # Start the rep at a moderate squat depth instead of waiting for a
        # very deep knee bend. The upward threshold requires standing again.
        if knee_angle < 125 and worst_knee < 135:
            stage = 'down'
        elif knee_angle > 155 and stage == 'down' and now - last_rep_time >= REP_COOLDOWN:
            count += 1
            last_rep_time = now
            stage = 'up'
            speak(f"Squat {count}")

    # ------------------ Push-up ------------------
    elif exercise_name == 'Push-ups':
        left_elbow = smooth_value('push_l_elbow', angle(LS, LE, LW), 5)
        right_elbow = smooth_value('push_r_elbow', angle(RS, RE, RW), 5)
        elbow_angle = bilateral_mean(left_elbow, right_elbow)
        worst_elbow = min(left_elbow, right_elbow)

        shoulder_mid = avg(LS, RS)
        hip_mid = avg(LH, RH)
        ankle_mid = avg(LA, RA)
        body_angle = smooth_value('push_body', angle(shoulder_mid, hip_mid, ankle_mid), 7)
        progress = interp_clip(elbow_angle, (170, 65))

        # A straight push-up is close to a shoulder-hip-ankle line. The
        # perpendicular hip offset catches both sagging and piking better
        # than a single angle threshold.
        hip_line_error = point_line_distance(hip_mid, shoulder_mid, ankle_mid) / max(torso, 0.08)
        line_score = interp_clip(body_angle, (135, 170), (0, 100))
        hip_score = interp_clip(hip_line_error, (0.025, PUSHUP['hip_sag_tolerance']), (100, 0))
        elbow_symmetry = interp_clip(abs(left_elbow - right_elbow), (8, 35), (100, 0))

        posture_score = line_score * 0.55 + hip_score * 0.30 + elbow_symmetry * 0.15
        if body_angle < 125 or hip_line_error > 0.13:
            posture_score = min(posture_score, 50)
        set_posture(posture_score)

        if elbow_angle < PUSHUP['down_threshold'] and worst_elbow < 115:
            stage = 'down'
        elif elbow_angle > PUSHUP['up_threshold'] and stage == 'down' and now - last_rep_time >= REP_COOLDOWN:
            count += 1
            last_rep_time = now
            stage = 'up'
            speak(f"Push-up {count}")

    # ------------------ Plank ------------------
    elif exercise_name == 'Plank':
        shoulder_mid = avg(LS, RS)
        hip_mid = avg(LH, RH)
        ankle_mid = avg(LA, RA)
        body_angle = smooth_value('plank_body', angle(shoulder_mid, hip_mid, ankle_mid), 7)

        # Use both the overall body angle and the hip's distance from the
        # shoulder-to-ankle line. This rejects both hip sag and excessive pike.
        hip_line_error = point_line_distance(hip_mid, shoulder_mid, ankle_mid) / max(torso, 0.08)
        line_score = interp_clip(body_angle, (135, 175), (0, 100))
        hip_score = interp_clip(hip_line_error, (0.02, 0.10), (100, 0))
        set_posture(line_score * 0.60 + hip_score * 0.40)

        if correct_posture:
            hold_start = hold_start or now
            progress = np.clip((now - hold_start) / PLANK['hold_seconds'] * 100, 0, 100)
        else:
            hold_start = 0
            progress = 0
            stage = None if stage == 'held' else stage

        # Plank is a timed hold; count one completed hold.
        if hold_start and now - hold_start >= PLANK['hold_seconds'] and stage != 'held':
            count += 1
            last_rep_time = now
            stage = 'held'
            speak(f"Plank complete {count}")

    # ------------------ Lunges ------------------
    elif exercise_name == 'Lunges':
        lk = smooth_value('lunge_lk', angle(LH, LK, LA), 5)
        rk = smooth_value('lunge_rk', angle(RH, RK, RA), 5)
        # The more-flexed knee is the working leg.
        knee_angle = min(lk, rk)
        working_left = lk <= rk
        work_knee = LK if working_left else RK
        work_ankle = LA if working_left else RA
        work_hip = LH if working_left else RH

        # Knee tracking: avoid the working knee drifting excessively ahead
        # of the ankle or collapsing far inward/outward.
        knee_ankle_dist = abs(work_knee[0] - work_ankle[0]) / max(dist(work_hip, work_ankle), 0.08)
        knee_track_ok = knee_ankle_dist <= 0.55

        # Keep the torso reasonably upright during the lunge.
        torso_angle = angle(avg(LS, RS), avg(LH, RH), avg(LK, RK))
        torso_ok = torso_angle >= 145

        # Penalize asymmetry instead of allowing one knee to hide a large
        # difference between the two legs.
        knee_symmetry = abs(lk - rk)
        symmetry_ok = knee_symmetry <= 45

        form_score = 100.0
        if not knee_track_ok:
            form_score -= 25
        if not torso_ok:
            form_score -= 25
        if not symmetry_ok:
            form_score -= 15
        if knee_angle < 75:
            form_score -= 10
        set_posture(form_score)
        progress = interp_clip(knee_angle, (170, LUNGE['down_threshold']))

        if knee_angle < LUNGE['down_threshold']:
            stage = 'down'
        elif (
            knee_angle > LUNGE['up_threshold']
            and stage == 'down'
            and now - last_rep_time >= REP_COOLDOWN
        ):
            count += 1
            last_rep_time = now
            stage = 'up'
            speak(f"Lunge {count}")

    # ------------------ Shoulder Press ------------------
    elif exercise_name == 'Shoulder Press':
        la = smooth_value('press_l', angle(LS, LE, LW), 5)
        ra = smooth_value('press_r', angle(RS, RE, RW), 5)
        elbow_angle = max(la, ra)

        # A genuine overhead press needs the wrists to travel above the
        # shoulders, not merely straightening the elbows in front of the body.
        left_overhead = LW[1] < LS[1] - 0.04
        right_overhead = RW[1] < RS[1] - 0.04
        overhead = left_overhead or right_overhead

        # Penalize a large left/right elbow mismatch and excessive torso lean.
        torso_angle = angle(avg(LS, RS), avg(LH, RH), avg(LK, RK))
        elbow_symmetry = abs(la - ra)
        torso_ok = torso_angle >= 145
        symmetry_ok = elbow_symmetry <= 40

        form_score = 100.0
        if not overhead and elbow_angle > 145:
            form_score -= 20
        if not torso_ok:
            form_score -= 30
        if not symmetry_ok:
            form_score -= 20

        set_posture(form_score)
        progress = interp_clip(elbow_angle, (80, 170))

        if overhead and elbow_angle > SHOULDER_PRESS['up_threshold']:
            stage = 'up'
        elif (
            elbow_angle < SHOULDER_PRESS['down_threshold']
            and stage == 'up'
            and now - last_rep_time >= REP_COOLDOWN
        ):
            count += 1
            last_rep_time = now
            stage = 'down'
            speak(f"Shoulder Press {count}")

    # ------------------ Glute Bridge ------------------
    elif exercise_name == 'Glute Bridge':
        hip_angle = smooth_value(
            'bridge_hip',
            angle(avg(LS, RS), avg(LH, RH), avg(LK, RK)),
            7
        )

        shoulder_mid = avg(LS, RS)
        hip_mid = avg(LH, RH)
        knee_mid = avg(LK, RK)

        # In the top position the shoulder-hip-knee chain should be close to
        # a straight line. This is more reliable than hip angle alone because
        # it catches both under-extension and exaggerated lumbar arching.
        bridge_line_angle = angle(shoulder_mid, hip_mid, knee_mid)
        hip_height = (knee_mid[1] - hip_mid[1]) / torso
        shoulder_hip_symmetry = abs((LH[1] - RH[1]) + (LK[1] - RK[1]))

        line_ok = bridge_line_angle >= 155
        height_ok = hip_height >= 0.18
        symmetry_ok = shoulder_hip_symmetry <= 0.18

        form_score = 100.0
        if not line_ok:
            form_score -= 25
        if not height_ok:
            form_score -= 25
        if not symmetry_ok:
            form_score -= 15

        set_posture(form_score)
        progress = interp_clip(hip_angle, (90, 175))

        if hip_angle > GLUTE_BRIDGE['up_threshold'] and height_ok:
            stage = 'up'
        elif (
            hip_angle < GLUTE_BRIDGE['down_threshold']
            and stage == 'up'
            and now - last_rep_time >= REP_COOLDOWN
        ):
            count += 1
            last_rep_time = now
            stage = 'down'
            speak(f"Glute Bridge {count}")

    # ------------------ Mountain Climbers ------------------
    elif exercise_name == 'Mountain Climbers':
        # Count only when the athlete is in a stable plank and one knee is
        # genuinely driven toward the chest. This prevents hip rocking from
        # being interpreted as repetitions.
        body_line = angle(avg(LS, RS), avg(LH, RH), avg(LA, RA))
        hip_to_shoulder = dist(avg(LS, RS), avg(LH, RH))
        hip_to_ankle = dist(avg(LH, RH), avg(LA, RA))
        body_ratio = hip_to_ankle / max(hip_to_shoulder, 0.08)
        left_drive = dist(LK, avg(LS, RS)) / torso
        right_drive = dist(RK, avg(LS, RS)) / torso
        left_up = left_drive < MOUNTAIN_CLIMBER['knee_threshold']
        right_up = right_drive < MOUNTAIN_CLIMBER['knee_threshold']
        plank_ok = body_line >= MOUNTAIN_CLIMBER['body_line_min'] and body_ratio > 1.05
        if plank_ok:
            progress = 100 if left_up or right_up else 15
        else:
            progress = 0
        posture_score = 100.0
        posture_score -= max(0, MOUNTAIN_CLIMBER['body_line_min'] - body_line) * 1.8
        posture_score -= max(0, 1.05 - body_ratio) * 120
        posture_score = np.clip(posture_score, 0, 100)
        set_posture(posture_score)
        if plank_ok and left_up and not right_up and stage != 'left':
            stage = 'left'
            if now - last_rep_time >= FAST_REP_COOLDOWN:
                count += 1; last_rep_time = now; speak(f"Mountain Climber {count}")
        elif plank_ok and right_up and not left_up and stage != 'right':
            stage = 'right'
            if now - last_rep_time >= FAST_REP_COOLDOWN:
                count += 1; last_rep_time = now; speak(f"Mountain Climber {count}")
        elif not left_up and not right_up:
            stage = 'neutral'

    # ------------------ Jumping Jacks ------------------
    elif exercise_name == 'Jumping Jacks':
        shoulder_mid = avg(LS, RS)
        wrist_height_ok = LW[1] < LS[1] and RW[1] < RS[1]
        feet_width = dist(LA, RA) / shoulder_width
        arm_width = dist(LW, RW) / shoulder_width
        open_pose = wrist_height_ok and feet_width >= JUMPING_JACKS['leg_open_threshold'] and arm_width >= 1.25
        closed_pose = feet_width < 1.05 and arm_width < 1.15
        posture_score = 100.0
        if not wrist_height_ok: posture_score -= 30
        posture_score -= max(0, JUMPING_JACKS['leg_open_threshold'] - feet_width) * 20
        set_posture(posture_score)
        progress = 100 if open_pose else (50 if closed_pose else 0)
        if open_pose:
            stage = 'open'
        elif closed_pose and stage == 'open' and now - last_rep_time >= REP_COOLDOWN:
            count += 1; last_rep_time = now; stage = 'closed'; speak(f"Jumping Jack {count}")

    # ------------------ High Knees ------------------
    elif exercise_name == 'High Knees':
        torso_angle = angle(avg(LS, RS), avg(LH, RH), avg(LK, RK))
        left_ratio = (LH[1] - LK[1]) / torso
        right_ratio = (RH[1] - RK[1]) / torso
        left_up = left_ratio < HIGH_KNEES['knee_up_threshold']
        right_up = right_ratio < HIGH_KNEES['knee_up_threshold']
        torso_ok = torso_angle >= HIGH_KNEES['torso_min']
        posture_score = interp_clip(torso_angle, (130, 175))
        set_posture(posture_score)
        progress = 100 if torso_ok and (left_up or right_up) else 0
        if torso_ok and left_up and not right_up and stage != 'left':
            stage = 'left'
            if now - last_rep_time >= FAST_REP_COOLDOWN:
                count += 1; last_rep_time = now; speak(f"High Knee {count}")
        elif torso_ok and right_up and not left_up and stage != 'right':
            stage = 'right'
            if now - last_rep_time >= FAST_REP_COOLDOWN:
                count += 1; last_rep_time = now; speak(f"High Knee {count}")
        elif not left_up and not right_up:
            stage = 'neutral'

    # ------------------ Side Lunges ------------------
    elif exercise_name == 'Side Lunges':
        lk = smooth_value('side_lk', angle(LH, LK, LA), 5)
        rk = smooth_value('side_rk', angle(RH, RK, RA), 5)
        working_left = lk < rk
        knee_angle = lk if working_left else rk
        working_knee, working_hip, working_ankle = (LK, LH, LA) if working_left else (RK, RH, RA)
        opposite_knee = RK if working_left else LK
        hip_mid = avg(LH, RH); shoulder_mid = avg(LS, RS)
        lateral = abs(hip_mid[0] - shoulder_mid[0]) / shoulder_width
        knee_track = abs(working_knee[0] - working_ankle[0]) / max(dist(working_knee, working_ankle), 0.08)
        torso_angle = angle(shoulder_mid, hip_mid, avg(working_knee, opposite_knee))
        posture_score = 100 - max(0, SIDE_LUNGE['down_threshold'] - knee_angle) * 0.6
        posture_score -= max(0, 0.10 - lateral) * 100
        posture_score -= max(0, knee_track - 0.55) * 80
        posture_score -= max(0, 145 - torso_angle) * 0.7
        set_posture(posture_score)
        progress = interp_clip(knee_angle, (170, SIDE_LUNGE['down_threshold'])) if lateral >= SIDE_LUNGE['hip_shift_threshold'] else 0
        if lateral >= SIDE_LUNGE['hip_shift_threshold'] and knee_angle < SIDE_LUNGE['down_threshold']:
            stage = 'down'
        elif knee_angle > SIDE_LUNGE['up_threshold'] and stage == 'down' and now - last_rep_time >= REP_COOLDOWN:
            count += 1; last_rep_time = now; stage = 'up'; speak(f"Side Lunge {count}")

    # ------------------ Side Leg Raises ------------------
    elif exercise_name == 'Side Leg Raises':
        left_raise = abs(LA[0] - LH[0]) / torso
        right_raise = abs(RA[0] - RH[0]) / torso
        left_raise = smooth_value('left_side_raise', left_raise, 5)
        right_raise = smooth_value('right_side_raise', right_raise, 5)
        raise_amount = max(left_raise, right_raise)
        hip_shift = abs(avg(LH, RH)[0] - avg(LS, RS)[0]) / shoulder_width
        torso_angle = angle(avg(LS, RS), avg(LH, RH), avg(LK, RK))
        posture_score = interp_clip(torso_angle, (120, 175)) - max(0, hip_shift - 0.16) * 120
        set_posture(posture_score)
        progress = interp_clip(raise_amount, (0.06, 0.40))
        if raise_amount > SIDE_LEG_RAISE['up_threshold'] and hip_shift < 0.25:
            stage = 'up'
        elif raise_amount < SIDE_LEG_RAISE['down_threshold'] and stage == 'up' and now - last_rep_time >= REP_COOLDOWN:
            count += 1; last_rep_time = now; stage = 'down'; speak(f"Side Leg Raise {count}")

    # ------------------ Wall Sit ------------------
    elif exercise_name == 'Wall Sit':
        lk = smooth_value('wall_lk', angle(LH, LK, LA), 7)
        rk = smooth_value('wall_rk', angle(RH, RK, RA), 7)
        knee_angle = (lk + rk) / 2
        torso_angle = angle(avg(LS, RS), avg(LH, RH), avg(LK, RK))
        knee_score = 100 - min(abs(knee_angle - WALL_SIT['knee_angle_target']) * 2.5, 100)
        torso_score = interp_clip(torso_angle, (125, 175))
        symmetry_score = max(0.0, 100 - abs(lk - rk) * 2.0)
        posture_score = 0.55 * knee_score + 0.30 * torso_score + 0.15 * symmetry_score
        set_posture(posture_score)
        if correct_posture:
            hold_start = hold_start or now
            progress = np.clip((now - hold_start) / WALL_SIT['hold_seconds'] * 100, 0, 100)
        else:
            progress = 0
        if hold_start and now - hold_start >= WALL_SIT['hold_seconds'] and stage != 'held':
            count += 1; last_rep_time = now; stage = 'held'; speak(f"Wall Sit complete {count}")

    # ------------------ Standing Knee-to-Elbow ------------------
    elif exercise_name == 'Standing Knee-to-Elbow':
        left_cross = dist(LK, RE) / torso
        right_cross = dist(RK, LE) / torso
        left_touch = left_cross < STANDING_KNEE_ELBOW['knee_up_threshold']
        right_touch = right_cross < STANDING_KNEE_ELBOW['knee_up_threshold']
        torso_angle = angle(avg(LS, RS), avg(LH, RH), avg(LK, RK))
        hip_shift = abs(avg(LH, RH)[0] - avg(LS, RS)[0]) / shoulder_width
        posture_score = interp_clip(torso_angle, (120, 175)) - max(0, hip_shift - STANDING_KNEE_ELBOW['max_lean']) * 180
        set_posture(posture_score)
        progress = 100 if (left_touch or right_touch) and torso_angle >= STANDING_KNEE_ELBOW['torso_min'] else 0
        if left_touch and not right_touch and stage != 'left':
            stage = 'left'
            if now - last_rep_time >= FAST_REP_COOLDOWN:
                count += 1; last_rep_time = now; speak(f"Knee to Elbow {count}")
        elif right_touch and not left_touch and stage != 'right':
            stage = 'right'
            if now - last_rep_time >= FAST_REP_COOLDOWN:
                count += 1; last_rep_time = now; speak(f"Knee to Elbow {count}")
        elif not left_touch and not right_touch:
            stage = 'neutral'

    # ------------------ Arm Circles ------------------
    elif exercise_name == 'Arm Circles':
        visible = min(lsv, lwv, rsv, rwv) >= ARM_CIRCLES['min_visibility']
        left_r = dist(LW, LS) / torso
        right_r = dist(RW, RS) / torso
        extended = left_r >= ARM_CIRCLES['min_arm_extension'] and right_r >= ARM_CIRCLES['min_arm_extension']
        if visible and extended:
            left_angle = np.degrees(np.arctan2(LW[1] - LS[1], LW[0] - LS[0]))
            right_angle = np.degrees(np.arctan2(RW[1] - RS[1], RW[0] - RS[0]))
            current_angle = np.unwrap([np.radians((left_angle + right_angle) / 2)])[0] * 180 / np.pi
            prev = state.get('circle_prev_angle')
            if prev is None:
                state['circle_prev_angle'] = current_angle
                state['circle_rotation'] = 0.0
                state['circle_direction'] = 0
            else:
                delta = current_angle - prev
                while delta > 180: delta -= 360
                while delta < -180: delta += 360
                if abs(delta) >= ARM_CIRCLES['min_delta']:
                    direction = 1 if delta > 0 else -1
                    if state.get('circle_direction', 0) == 0:
                        state['circle_direction'] = direction
                    if direction == state.get('circle_direction'):
                        state['circle_rotation'] = state.get('circle_rotation', 0.0) + delta
                    else:
                        state['circle_rotation'] *= 0.85
                state['circle_prev_angle'] = current_angle
            rotation = abs(state.get('circle_rotation', 0.0))
            progress = interp_clip(rotation, (0, ARM_CIRCLES['rotation_target']))
            if rotation >= ARM_CIRCLES['rotation_target'] and now - last_rep_time >= ARM_CIRCLES['cooldown']:
                count += 1; last_rep_time = now; state['circle_rotation'] = 0.0; state['circle_direction'] = 0; speak(f"Arm Circle {count}")
            set_posture(100)
        else:
            state.pop('circle_prev_angle', None)
            state['circle_rotation'] = 0.0
            state['circle_direction'] = 0
            progress = 0
            set_posture(65)

    else:
        # Unknown exercise: do not crash; keep telemetry usable.
        set_posture(0)
        progress = 0

    # Keep progress smooth and bounded for the UI.
    progress = float(np.clip(progress, 0, 100))
    state['stage'] = stage
    state['count'] = int(count)
    state['last_rep_time'] = last_rep_time
    state['hold_start'] = hold_start
    state['posture_ok_since'] = posture_ok_since
    state['low_visibility'] = all_vis < min_visibility

    return state, bool(correct_posture), progress, float(posture_score)

# ------------------ Camera and main loop

def run_camera_and_track(exercise_name, height, weight):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Could not open camera.")
        return

    # smoother holds per-joint buffers
    smoother = LandmarkSmoother()
    state = {'stage': None, 'count': 0, 'last_rep_time': 0, 'hold_start': 0, 'posture_ok_since': None}

    window_name = "AI Fitness Assistant"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    speak(f"Starting {exercise_name}. Get ready!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        now = time.time()

        correct_posture = False
        progress = 0.0
        posture_score = 0.0

        if result.pose_landmarks:
            # feed landmarks into smoother
            for i, lm in enumerate(result.pose_landmarks.landmark):
                smoother.add(i, lm.x, lm.y, lm.visibility)

            state, correct_posture, progress, posture_score = exercise_logic(
                exercise_name, result.pose_landmarks.landmark, mp_pose, smoother, state, now
            )

            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # announce posture change events (debounced)
            last_posture_state = state.get('_last_posture_state')
            if correct_posture != last_posture_state:
                state['_last_posture_state'] = correct_posture
                speak("Good posture." if correct_posture else "Adjust your posture.")
                state['_last_warning_time'] = now

        # Draw Progress bar (left)
        cv2.rectangle(frame, (50, 150), (80, 400), (255, 255, 255), 3)
        filled_y1 = int(400 - (progress * 2.5))
        color_move = (0, 255, 0) if correct_posture else (0, 0, 255)
        cv2.rectangle(frame, (50, filled_y1), (80, 400), color_move, -1)
        cv2.putText(frame, "Move", (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Posture bar (right)
        cv2.rectangle(frame, (680, 150), (710, 400), (255, 255, 255), 3)
        filled_y2 = int(400 - (posture_score * 2.5))
        color_form = (255, 255, 0) if posture_score > 80 else (0, 0, 255)
        cv2.rectangle(frame, (680, filled_y2), (710, 400), color_form, -1)
        cv2.putText(frame, "Form", (670, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Text info
        bmi = weight / (height ** 2)
        cv2.putText(frame, f"BMI: {bmi:.1f}", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Exercise: {exercise_name}", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Count: {state.get('count',0)}", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Posture: {'Good' if correct_posture else 'Bad'}",
                    (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if correct_posture else (0, 0, 255), 2)
        cv2.putText(frame, f"Form Score: {int(posture_score)}", (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow(window_name, frame)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    speak("Workout complete. Great job!")

# ------------------ GUI (cleaned up) ------------------

def start_tracking():
    try:
        height = float(height_var.get())
        weight = float(weight_var.get())
        if height <= 0 or weight <= 0:
            result_label.config(text="❌ Enter positive values!")
            return

        bmi = weight / (height ** 2)
        exercises = recommend_exercise(bmi)
        result_label.config(text=f"BMI: {bmi:.2f}\nRecommended: {', '.join(exercises)}")

        selected_exercise = exercise_var.get()
        threading.Thread(target=run_camera_and_track,
                         args=(selected_exercise, height, weight), daemon=True).start()
    except ValueError:
        result_label.config(text="❌ Enter valid numbers!")


# ------------------ BMI recommendation & unchanged helpers ------------------

def recommend_exercise(bmi):
    if bmi < 18.5:
        return ["Push-ups", "Plank", "Bicep Curl","Shoulder Press" ,"Glute Bridge","Arm Circles"]
    elif bmi < 25:
        return ["Squats", "Lunges", "Plank", "Bicep Curl","Push-ups" , "Shoulder Press","Glute Bridge", "Mountain Climbers","Jumping Jacks" , "High Knees","Side Lunges","Side Leg Raises","Standing Knee-to-Elbow","Arm Circles"]
    else:
        return ["Walking", "Stretching", "Push-ups", "Plank","Mountain Climbers", "Glute Bridge","Squats", "Lunges","Jumping Jacks", "High Knees","Side Lunges","Side Leg Raises","Standing Knee-to-Elbow","Arm Circles"]       


# ------------------ UI Setup ------------------
# Only launch the Tkinter GUI when running this file directly (not when
# imported by the FastAPI backend).
if __name__ == "__main__":
    root = tk.Tk()
    root.title("AI Fitness Assistant")
    root.geometry("460x480")
    root.configure(bg="#0f1720")

    tk.Label(root, text="AI Fitness Assistant", fg="white", bg="#0f1720",
             font=("Helvetica", 18, "bold")).pack(pady=12)

    for label_text in ["Height (m):", "Weight (kg):"]:
        tk.Label(root, text=label_text, bg="#0f1720", fg="white").pack(anchor="w", padx=20)

    # use StringVar for entries (more robust than root.children hack)
    height_var = tk.StringVar()
    weight_var = tk.StringVar()
    height_entry = tk.Entry(root, textvariable=height_var)
    height_entry.pack(fill="x", padx=20, pady=(0, 8))
    weight_entry = tk.Entry(root, textvariable=weight_var)
    weight_entry.pack(fill="x", padx=20, pady=(0, 8))

    tk.Label(root, text="Select Exercise:", bg="#0f1720", fg="white").pack(anchor="w", padx=20)
    exercise_var = tk.StringVar(value="Bicep Curl")
    exercise_dropdown = ttk.Combobox(root, textvariable=exercise_var, state="readonly",
                                     values=("Bicep Curl", "Squats", "Push-ups", "Plank","Lunges","Shoulder Press","Glute Bridge","Mountain Climbers", "Jumping Jacks", "High Knees", "Side Lunges", "Side Leg Raises","Wall Sit","Standing Knee-to-Elbow","Arm Circles"))
    exercise_dropdown.pack(fill="x", padx=20, pady=6)

    tk.Button(root, text="Start Exercise", command=start_tracking,
              bg="#00a3e0", fg="white", font=("Arial", 11, "bold")).pack(pady=12)

    result_label = tk.Label(root, text="", bg="#0f1720", fg="lightgreen", font=("Arial", 11))
    result_label.pack(pady=6)

    tk.Label(root, text="Tip: Press 'q' or close the window to stop.",
             bg="#0f1720", fg="#9fbddc").pack(pady=10)

    root.mainloop()