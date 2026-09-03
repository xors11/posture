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
REP_COOLDOWN = 0.9             # minimum seconds between counted reps
VOICE_BACKGROUND = True        # speak in background thread to avoid blocking

# Angle thresholds / mapping (kept together so tuning is easy)
BICEP = {
    "back_angle_range": (150, 180),   # map to posture score
    "back_ok": 80,
    "elbow_range": (160, 40),         # (straight, flexed)
    "progress_down_thresh": 15,
    "progress_up_thresh": 90
}

SQUAT = {
    "back_angle_range": (140, 180),
    "back_ok": 78,
    "knee_range": (170, 60),
    "progress_down_thresh": 85,
    "progress_up_thresh": 20
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
    "knee_threshold": 0.55,
    "min_visibility": 0.7
}

JUMPING_JACKS = {
    "arm_up_threshold": 0.75,
    "leg_open_threshold": 0.35,
    "min_visibility": 0.7
}

HIGH_KNEES = {
    "knee_up_threshold": 0.75,
    "min_visibility": 0.7
}

SIDE_LUNGE = {
    "down_threshold": 110,
    "up_threshold": 160,
    "min_visibility": 0.7,
    "hip_shift_threshold": 0.12
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
    "min_visibility": 0.7
}

ARM_CIRCLES = {
    "circle_radius": 0.18,
    "min_visibility": 0.7,
    "cooldown": 0.5
}




# ------------------ Speech Engine (non-blocking wrapper) ------------------
engine = pyttsx3.init()
engine.setProperty('rate', 170)

_speech_lock = threading.Lock()

def speak(text):
    if not text:
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
    out = np.interp(value, src_range, dst_range)
    return float(np.clip(out, dst_range[0], dst_range[1]))

# ------------------ Side selection improved: choose the most visible side or both

def side_and_visibility(landmarks, mp_pose):
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

    left_vis = np.mean([landmarks[j.value].visibility for j in left_joints])
    right_vis = np.mean([landmarks[j.value].visibility for j in right_joints])

    # if both visible, return BOTH so logic can choose best per-joint
    if left_vis >= MIN_VISIBILITY and right_vis >= MIN_VISIBILITY:
        return "BOTH", max(left_vis, right_vis)
    side = "LEFT" if left_vis >= right_vis else "RIGHT"
    return side, max(left_vis, right_vis)

# ------------------ Exercise logic (cleaner, more robust)

def exercise_logic(exercise_name, landmarks, mp_pose, smoother,
                   state, now,
                   min_visibility=MIN_VISIBILITY):
    """state is a dict that carries across frames: {stage, count, last_rep_time, hold_start, posture_ok_since, angle_buffers}
    returns updated state and telemetry (correct_posture, progress, posture_score)
    """
    # prepare state defaults
    stage = state.get('stage')  # 'down' or 'up' or None
    count = state.get('count', 0)
    last_rep_time = state.get('last_rep_time', 0)
    hold_start = state.get('hold_start', 0)
    posture_ok_since = state.get('posture_ok_since', None)

    # choose side
    side, visibility = side_and_visibility(landmarks, mp_pose)
    if visibility < min_visibility:
        # insufficient visibility: consider as no detection
        return state, False, 0.0, 0.0

    # helper to read smoothed landmark for a mediapipe index
    def L(idx):
        sm = smoother.smoothed(idx)
        if sm is None:
            lm = landmarks[idx]
            return lm.x, lm.y, lm.visibility
        return sm

    # dynamic indices for whichever side(s) available
    def get_joint(name, which='LEFT'):
        enum_map = {
            'LEFT': {
                'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                'elbow': mp_pose.PoseLandmark.LEFT_ELBOW.value,
                'wrist': mp_pose.PoseLandmark.LEFT_WRIST.value,
                'hip': mp_pose.PoseLandmark.LEFT_HIP.value,
                'knee': mp_pose.PoseLandmark.LEFT_KNEE.value,
                'ankle': mp_pose.PoseLandmark.LEFT_ANKLE.value
            },
            'RIGHT': {
                'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                'wrist': mp_pose.PoseLandmark.RIGHT_WRIST.value,
                'hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
                'knee': mp_pose.PoseLandmark.RIGHT_KNEE.value,
                'ankle': mp_pose.PoseLandmark.RIGHT_ANKLE.value
            }
        }
        return L(enum_map[which][name])

    # If BOTH sides are visible, compute both and pick the one with higher wrist visibility for arm exercises or higher knee visibility for squats
    chosen_side = None
    if side == 'BOTH':
        left_wrist_vis = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility
        right_wrist_vis = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility
        left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
        right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

        if exercise_name in ('Bicep Curl', 'Push-ups', 'Shoulder Press','Arm Circles'):
            chosen_side = 'LEFT' if left_wrist_vis >= right_wrist_vis else 'RIGHT'
        elif exercise_name in ('Squats' , 'Lunges', 'Glute Bridge', 'Mountain Climbers', 'Jumping Jacks', 'High Knees','Side Lunges','Side Leg Raises','Wall Sit','Standing Knee-to-Elbow'):
            chosen_side = 'LEFT' if left_knee_vis >= right_knee_vis else 'RIGHT'
        else:
            chosen_side = 'LEFT' if left_wrist_vis >= right_wrist_vis else 'RIGHT'
    else:
        chosen_side = side

    # fetch joints for chosen_side
    shoulder_x, shoulder_y, shoulder_vis = get_joint('shoulder', chosen_side)
    elbow_x, elbow_y, elbow_vis = get_joint('elbow', chosen_side)
    wrist_x, wrist_y, wrist_vis = get_joint('wrist', chosen_side)
    hip_x, hip_y, hip_vis = get_joint('hip', chosen_side)
    knee_x, knee_y, knee_vis = get_joint('knee', chosen_side)
    ankle_x, ankle_y, ankle_vis = get_joint('ankle', chosen_side)

    shoulder = (shoulder_x, shoulder_y)
    elbow = (elbow_x, elbow_y)
    wrist = (wrist_x, wrist_y)
    hip = (hip_x, hip_y)
    knee = (knee_x, knee_y)
    ankle = (ankle_x, ankle_y)

    correct_posture = False
    progress = 0.0
    posture_score = 0.0

    if exercise_name == 'Bicep Curl':
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        back_angle = calculate_angle(shoulder, hip, knee)
        posture_score = interp_clip(back_angle, BICEP['back_angle_range'])
        correct_posture = posture_score > BICEP['back_ok']
        progress = interp_clip(elbow_angle, BICEP['elbow_range'])

        # smoothing the progress using a small buffer inside state
        buf = state.setdefault('angle_buf', deque(maxlen=SMOOTHING_WINDOW))
        buf.append(progress)
        smooth_progress = float(np.mean(buf))

        # posture_ok_since logic: only set once when posture becomes ok, clear when bad
        if correct_posture:
            if posture_ok_since is None:
                posture_ok_since = now
        else:
            posture_ok_since = None

        posture_ready = posture_ok_since is not None and (now - posture_ok_since) > POSTURE_HOLD_SEC

        # stage transitions (clean & debounced)
        if posture_ready:
            if smooth_progress < BICEP['progress_down_thresh']:
                stage = 'down'
            if smooth_progress > BICEP['progress_up_thresh'] and stage == 'down' and (now - last_rep_time) > REP_COOLDOWN:
                count += 1
                last_rep_time = now
                stage = 'up'
                speak(f"Repetition {count}")

        progress = smooth_progress

    elif exercise_name == "Arm Circles":

        left_shoulder = get_joint("shoulder", "LEFT")
        left_wrist = get_joint("wrist", "LEFT")

        right_shoulder = get_joint("shoulder", "RIGHT")
        right_wrist = get_joint("wrist", "RIGHT")

        visible = (
            left_shoulder[2] >= ARM_CIRCLES["min_visibility"]
            and left_wrist[2] >= ARM_CIRCLES["min_visibility"]
            and right_shoulder[2] >= ARM_CIRCLES["min_visibility"]
            and right_wrist[2] >= ARM_CIRCLES["min_visibility"]
        )

        # Wrist position relative to each shoulder.
        left_dx = left_wrist[0] - left_shoulder[0]
        left_dy = left_wrist[1] - left_shoulder[1]

        right_dx = right_wrist[0] - right_shoulder[0]
        right_dy = right_wrist[1] - right_shoulder[1]

        left_radius = np.sqrt(
            left_dx ** 2 + left_dy ** 2
        )

        right_radius = np.sqrt(
            right_dx ** 2 + right_dy ** 2
        )

        arms_extended = (
            left_radius >= ARM_CIRCLES["circle_radius"]
            and right_radius >= ARM_CIRCLES["circle_radius"]
        )

        # Track wrist direction around the shoulder.
        left_angle = np.degrees(
            np.arctan2(left_dy, left_dx)
        )

        right_angle = np.degrees(
            np.arctan2(right_dy, right_dx)
        )

        if visible and arms_extended:

            if "circle_prev_angle" not in state:
                state["circle_prev_angle"] = (
                    left_angle + right_angle
                ) / 2
                state["circle_rotation"] = 0

            current_angle = (
                left_angle + right_angle
            ) / 2

            angle_delta = current_angle - state["circle_prev_angle"]

            # Handle angle wrap-around.
            if angle_delta > 180:
                angle_delta -= 360
            elif angle_delta < -180:
                angle_delta += 360

            state["circle_rotation"] += angle_delta
            state["circle_prev_angle"] = current_angle

            if abs(state["circle_rotation"]) >= 300:

                if now - last_rep_time > ARM_CIRCLES["cooldown"]:
                    count += 1
                    last_rep_time = now
                    state["circle_rotation"] = 0
                    speak(f"Arm Circle {count}")

        else:
            state.pop("circle_prev_angle", None)
            state["circle_rotation"] = 0

        progress = interp_clip(
            abs(state.get("circle_rotation", 0)),
            (0, 300)
        )

        correct_posture = (
            visible
            and arms_extended
        )

        posture_score = 100 if correct_posture else 70
    
    elif exercise_name == "Standing Knee-to-Elbow":

        left_hip = get_joint("hip", "LEFT")
        left_knee = get_joint("knee", "LEFT")
        left_elbow = get_joint("elbow", "LEFT")

        right_hip = get_joint("hip", "RIGHT")
        right_knee = get_joint("knee", "RIGHT")
        right_elbow = get_joint("elbow", "RIGHT")

        visible = (
            left_hip[2] >= STANDING_KNEE_ELBOW["min_visibility"]
            and left_knee[2] >= STANDING_KNEE_ELBOW["min_visibility"]
            and left_elbow[2] >= STANDING_KNEE_ELBOW["min_visibility"]
            and right_hip[2] >= STANDING_KNEE_ELBOW["min_visibility"]
            and right_knee[2] >= STANDING_KNEE_ELBOW["min_visibility"]
            and right_elbow[2] >= STANDING_KNEE_ELBOW["min_visibility"]
        )

        left_leg_length = np.linalg.norm(
            np.array(left_hip[:2]) - np.array(left_knee[:2])
        )

        right_leg_length = np.linalg.norm(
            np.array(right_hip[:2]) - np.array(right_knee[:2])
        )

        left_knee_height = (
            abs(left_knee[1] - left_hip[1])
            / max(left_leg_length, 1e-6)
        )

        right_knee_height = (
            abs(right_knee[1] - right_hip[1])
            / max(right_leg_length, 1e-6)
        )

        left_knee_up = (
            left_knee_height
            < STANDING_KNEE_ELBOW["knee_up_threshold"]
        )

        right_knee_up = (
            right_knee_height
            < STANDING_KNEE_ELBOW["knee_up_threshold"]
        )

        left_elbow_knee_distance = np.linalg.norm(
            np.array(left_elbow[:2])
            - np.array(left_knee[:2])
        )

        right_elbow_knee_distance = np.linalg.norm(
            np.array(right_elbow[:2])
            - np.array(right_knee[:2])
        )

        knee_to_elbow_threshold = 0.25

        left_touch = (
            left_knee_up
            and left_elbow_knee_distance
            < knee_to_elbow_threshold
        )

        right_touch = (
            right_knee_up
            and right_elbow_knee_distance
            < knee_to_elbow_threshold
        )

        if visible:

            if left_touch and stage != "left":
                stage = "left"

                if now - last_rep_time > REP_COOLDOWN:
                    count += 1
                    last_rep_time = now
                    speak(f"Knee to Elbow {count}")

            elif right_touch and stage != "right":
                stage = "right"

                if now - last_rep_time > REP_COOLDOWN:
                    count += 1
                    last_rep_time = now
                    speak(f"Knee to Elbow {count}")

        progress = 100 if (left_touch or right_touch) else 0

        correct_posture = (
            visible
            and not (left_knee_up and right_knee_up)
        )

        posture_score = 100 if correct_posture else 70
    
    elif exercise_name == "Wall Sit":

        left_hip = get_joint("hip", "LEFT")
        left_knee = get_joint("knee", "LEFT")
        left_ankle = get_joint("ankle", "LEFT")

        right_hip = get_joint("hip", "RIGHT")
        right_knee = get_joint("knee", "RIGHT")
        right_ankle = get_joint("ankle", "RIGHT")

        visible = (
            left_hip[2] >= WALL_SIT["min_visibility"]
            and left_knee[2] >= WALL_SIT["min_visibility"]
            and left_ankle[2] >= WALL_SIT["min_visibility"]
            and right_hip[2] >= WALL_SIT["min_visibility"]
            and right_knee[2] >= WALL_SIT["min_visibility"]
            and right_ankle[2] >= WALL_SIT["min_visibility"]
        )

        left_knee_angle = calculate_angle(
            left_hip[:2],
            left_knee[:2],
            left_ankle[:2]
        )

        right_knee_angle = calculate_angle(
            right_hip[:2],
            right_knee[:2],
            right_ankle[:2]
        )

        target = WALL_SIT["knee_angle_target"]
        tolerance = WALL_SIT["knee_angle_tolerance"]

        left_correct = (
            target - tolerance
            <= left_knee_angle
            <= target + tolerance
        )

        right_correct = (
            target - tolerance
            <= right_knee_angle
            <= target + tolerance
        )

        correct_position = (
            left_correct
            and right_correct
        )

        if visible:

            if correct_position:

                if stage != "holding":
                    stage = "holding"
                    hold_start_time = now

                hold_time = now - hold_start_time

                if hold_time >= WALL_SIT["hold_seconds"]:
                    if now - last_rep_time > REP_COOLDOWN:
                        count += 1
                        last_rep_time = now
                        stage = "completed"
                        speak(f"Wall Sit {count}")

            else:
                stage = "ready"
                hold_start_time = now

        else:
            stage = "ready"
            hold_start_time = now

        if visible and correct_position:
            hold_time = now - hold_start_time
            progress = interp_clip(
                hold_time,
                (0, WALL_SIT["hold_seconds"])
            )
        else:
            progress = 0

        correct_posture = visible and correct_position
        posture_score = 100 if correct_posture else 70
    
    elif exercise_name == "Side Leg Raises":

        left_hip = get_joint("hip", "LEFT")
        left_knee = get_joint("knee", "LEFT")
        left_ankle = get_joint("ankle", "LEFT")

        right_hip = get_joint("hip", "RIGHT")
        right_knee = get_joint("knee", "RIGHT")
        right_ankle = get_joint("ankle", "RIGHT")

        visible = (
            left_hip[2] >= SIDE_LEG_RAISE["min_visibility"]
            and left_knee[2] >= SIDE_LEG_RAISE["min_visibility"]
            and left_ankle[2] >= SIDE_LEG_RAISE["min_visibility"]
            and right_hip[2] >= SIDE_LEG_RAISE["min_visibility"]
            and right_knee[2] >= SIDE_LEG_RAISE["min_visibility"]
            and right_ankle[2] >= SIDE_LEG_RAISE["min_visibility"]
        )

        hip_width = abs(left_hip[0] - right_hip[0])
        hip_width = max(hip_width, 1e-6)

        left_leg_offset = abs(left_ankle[0] - left_hip[0]) / hip_width
        right_leg_offset = abs(right_ankle[0] - right_hip[0]) / hip_width

        leg_offset = max(
            left_leg_offset,
            right_leg_offset
        )

        left_knee_angle = calculate_angle(
            left_hip[:2],
            left_knee[:2],
            left_ankle[:2]
        )

        right_knee_angle = calculate_angle(
            right_hip[:2],
            right_knee[:2],
            right_ankle[:2]
        )

        knees_straight = (
            left_knee_angle >= SIDE_LEG_RAISE["max_knee_bend"]
            and right_knee_angle >= SIDE_LEG_RAISE["max_knee_bend"]
        )

        leg_raised = (
            leg_offset >= SIDE_LEG_RAISE["up_threshold"]
            and knees_straight
        )

        leg_lowered = (
            leg_offset <= SIDE_LEG_RAISE["down_threshold"]
        )

        if visible:

            if leg_raised:
                stage = "up"

            elif leg_lowered and stage == "up":
                if now - last_rep_time > REP_COOLDOWN:
                    count += 1
                    last_rep_time = now
                    stage = "down"
                    speak(f"Side Leg Raise {count}")

        progress = interp_clip(
            leg_offset,
            (
                SIDE_LEG_RAISE["down_threshold"],
                SIDE_LEG_RAISE["up_threshold"]
            )
        )

        correct_posture = (
            visible
            and knees_straight
        )

        posture_score = 100 if correct_posture else 70
        
    
    elif exercise_name == "Side Lunges":

        left_hip = get_joint("hip", "LEFT")
        left_knee = get_joint("knee", "LEFT")
        left_ankle = get_joint("ankle", "LEFT")

        right_hip = get_joint("hip", "RIGHT")
        right_knee = get_joint("knee", "RIGHT")
        right_ankle = get_joint("ankle", "RIGHT")

        left_shoulder = get_joint("shoulder", "LEFT")
        right_shoulder = get_joint("shoulder", "RIGHT")

        visible = (
            left_hip[2] >= SIDE_LUNGE["min_visibility"]
            and left_knee[2] >= SIDE_LUNGE["min_visibility"]
            and left_ankle[2] >= SIDE_LUNGE["min_visibility"]
            and right_hip[2] >= SIDE_LUNGE["min_visibility"]
            and right_knee[2] >= SIDE_LUNGE["min_visibility"]
            and right_ankle[2] >= SIDE_LUNGE["min_visibility"]
        )

        left_knee_angle = calculate_angle(
            left_hip[:2],
            left_knee[:2],
            left_ankle[:2]
        )

        right_knee_angle = calculate_angle(
            right_hip[:2],
            right_knee[:2],
            right_ankle[:2]
        )

        # Determine which leg is currently bending.
        if left_knee_angle < right_knee_angle:
            active_side = "LEFT"
            active_knee_angle = left_knee_angle
        else:
            active_side = "RIGHT"
            active_knee_angle = right_knee_angle

        # Detect lateral movement of the hips.
        shoulder_mid_x = (
            left_shoulder[0] + right_shoulder[0]
        ) / 2

        hip_mid_x = (
            left_hip[0] + right_hip[0]
        ) / 2

        hip_shift = abs(hip_mid_x - shoulder_mid_x)

        # Normalize hip movement relative to shoulder width.
        shoulder_width = abs(
            left_shoulder[0] - right_shoulder[0]
        )

        shoulder_width = max(shoulder_width, 1e-6)

        normalized_hip_shift = hip_shift / shoulder_width

        lateral_movement = (
            normalized_hip_shift >= SIDE_LUNGE["hip_shift_threshold"]
        )

        good_form = (
            active_knee_angle >= 70
            and active_knee_angle <= 180
            and lateral_movement
        )

        if visible:

            # Down position
            if (
                active_knee_angle < SIDE_LUNGE["down_threshold"]
                and lateral_movement
                ):
                stage = "down"

            # Return to standing = one repetition
            elif (
                active_knee_angle > SIDE_LUNGE["up_threshold"]
                and stage == "down"
                and now - last_rep_time > REP_COOLDOWN
            ):
                count += 1
                last_rep_time = now
                stage = "up"
                speak(f"Side Lunge {count}")

        progress = interp_clip(
            active_knee_angle,
            (
                SIDE_LUNGE["down_threshold"],
                SIDE_LUNGE["up_threshold"]
            )
        )

        correct_posture = visible and good_form

        posture_score = (
            100 if correct_posture else 70
        )
    
    elif exercise_name == 'Squats':
        knee_angle = calculate_angle(hip, knee, ankle)
        back_angle = calculate_angle(shoulder, hip, knee)
        posture_score = interp_clip(back_angle, SQUAT['back_angle_range'])
        correct_posture = posture_score > SQUAT['back_ok']
        progress = interp_clip(knee_angle, SQUAT['knee_range'])

        buf = state.setdefault('angle_buf', deque(maxlen=SMOOTHING_WINDOW))
        buf.append(progress)
        smooth_progress = float(np.mean(buf))

        if correct_posture:
            if posture_ok_since is None:
                posture_ok_since = now
        else:
            posture_ok_since = None

        posture_ready = posture_ok_since is not None and (now - posture_ok_since) > POSTURE_HOLD_SEC

        if posture_ready:
            # For squats, progress ~100 when standing, ~0 when deep squat depending on mapping above
            # Using mapped thresholds
            if smooth_progress > SQUAT['progress_down_thresh']:
                stage = 'down'  # standing
            if smooth_progress < SQUAT['progress_up_thresh'] and stage == 'down' and (now - last_rep_time) > REP_COOLDOWN:
                count += 1
                last_rep_time = now
                stage = 'up'
                speak(f"Squat {count}")

        progress = smooth_progress

    
    elif exercise_name == "Lunges":

        left_knee = calculate_angle(
            get_joint("hip", "LEFT")[:2],
            get_joint("knee", "LEFT")[:2],
            get_joint("ankle", "LEFT")[:2]
        )

        right_knee = calculate_angle(
            get_joint("hip", "RIGHT")[:2],
            get_joint("knee", "RIGHT")[:2],
            get_joint("ankle", "RIGHT")[:2]
        )

        side = "LEFT" if left_knee <= right_knee else "RIGHT"

        hip = get_joint("hip", side)
        knee = get_joint("knee", side)
        ankle = get_joint("ankle", side)
        shoulder = get_joint("shoulder", side)

        knee_angle = left_knee if side == "LEFT" else right_knee

        visible = (
            hip[2] > LUNGE["min_visibility"]
            and knee[2] > LUNGE["min_visibility"]
            and ankle[2] > LUNGE["min_visibility"]
        )

        torso_angle = calculate_angle(
            shoulder[:2],
            hip[:2],
            knee[:2]
        )

        shoulder_mid = (
            get_joint("shoulder", "LEFT")[0]
            + get_joint("shoulder", "RIGHT")[0]
        ) / 2

        hip_mid = (
            get_joint("hip", "LEFT")[0]
            + get_joint("hip", "RIGHT")[0]
        ) / 2

        balanced = abs(
            shoulder_mid - hip_mid
        ) <= LUNGE["balance_tolerance"]

        correct_posture = (
            visible
            and torso_angle > 140
            and balanced
        )

        progress = interp_clip(
            knee_angle,
            (
                LUNGE["down_threshold"],
                LUNGE["up_threshold"]
            )
        )

        if visible:

            if knee_angle < LUNGE["down_threshold"]:
                stage = "down"

            elif (
                knee_angle > LUNGE["up_threshold"]
                and stage == "down"
                and now - last_rep_time > REP_COOLDOWN
            ):
                count += 1
                stage = "up"
                last_rep_time = now
                speak(f"Lunge {count}")

        posture_score = interp_clip(
            torso_angle,
            (140, 180)
        )
    elif exercise_name == "High Knees":

        left_hip = get_joint("hip", "LEFT")
        left_knee = get_joint("knee", "LEFT")
        left_ankle = get_joint("ankle", "LEFT")

        right_hip = get_joint("hip", "RIGHT")
        right_knee = get_joint("knee", "RIGHT")
        right_ankle = get_joint("ankle", "RIGHT")

        visible = (
            left_hip[2] >= HIGH_KNEES["min_visibility"]
            and left_knee[2] >= HIGH_KNEES["min_visibility"]
            and left_ankle[2] >= HIGH_KNEES["min_visibility"]
            and right_hip[2] >= HIGH_KNEES["min_visibility"]
            and right_knee[2] >= HIGH_KNEES["min_visibility"]
            and right_ankle[2] >= HIGH_KNEES["min_visibility"]
        )

        # Vertical distance between hip and knee.
        # Smaller normalized distance means the knee is raised.
        left_leg_length = np.linalg.norm(
            np.array(left_hip[:2]) - np.array(left_ankle[:2])
        )

        right_leg_length = np.linalg.norm(
            np.array(right_hip[:2]) - np.array(right_ankle[:2])
        )

        left_knee_height = (
            abs(left_knee[1] - left_hip[1])
            / max(left_leg_length, 1e-6)
        )

        right_knee_height = (
            abs(right_knee[1] - right_hip[1])
            / max(right_leg_length, 1e-6)
        )

        left_knee_up = (
            left_knee_height < HIGH_KNEES["knee_up_threshold"]
        )

        right_knee_up = (
            right_knee_height < HIGH_KNEES["knee_up_threshold"]
        )

        if visible:

            # Alternate between left and right knee.
            if left_knee_up and stage != "left":

                stage = "left"

                if now - last_rep_time > REP_COOLDOWN:
                    count += 1
                    last_rep_time = now
                    speak(f"High Knee {count}")

            elif right_knee_up and stage != "right":

                stage = "right"

                if now - last_rep_time > REP_COOLDOWN:
                    count += 1
                    last_rep_time = now
                    speak(f"High Knee {count}")

        progress = 100 if (left_knee_up or right_knee_up) else 0

        correct_posture = (
            visible
            and not (left_knee_up and right_knee_up)
        )

        posture_score = 100 if correct_posture else 70
        
    elif exercise_name == "Jumping Jacks":

        left_wrist = get_joint("wrist", "LEFT")
        right_wrist = get_joint("wrist", "RIGHT")
        left_ankle = get_joint("ankle", "LEFT")
        right_ankle = get_joint("ankle", "RIGHT")

        visible = (
            left_wrist[2] >= JUMPING_JACKS["min_visibility"]
            and right_wrist[2] >= JUMPING_JACKS["min_visibility"]
            and left_ankle[2] >= JUMPING_JACKS["min_visibility"]
            and right_ankle[2] >= JUMPING_JACKS["min_visibility"]
        )

        # Normalize body measurements using shoulder width
        shoulder_width = abs(
            get_joint("shoulder", "LEFT")[0]
            - get_joint("shoulder", "RIGHT")[0]
        )

        shoulder_width = max(shoulder_width, 1e-6)

        # Hands above shoulder level
        hands_up = (
            left_wrist[1] < get_joint("shoulder", "LEFT")[1]
            and right_wrist[1] < get_joint("shoulder", "RIGHT")[1]
        )

        # Legs opened wider than normal stance
        ankle_distance = abs(left_ankle[0] - right_ankle[0])
        legs_open = (
            ankle_distance / shoulder_width
            > JUMPING_JACKS["leg_open_threshold"]
        )

        jumping_position = hands_up and legs_open

        if visible:

            # Open position
            if jumping_position:
                stage = "open"

            # Closed position after open = one repetition
            elif not jumping_position and stage == "open":
                if now - last_rep_time > REP_COOLDOWN:
                    count += 1
                    last_rep_time = now
                    stage = "closed"
                    speak(f"Jumping Jack {count}")

        progress = 100 if jumping_position else 0

        correct_posture = (
            visible
            and not (
                hands_up and not legs_open
            )
        )

        posture_score = 100 if correct_posture else 70
    
    elif exercise_name == "Shoulder Press":

        elbow_angle = calculate_angle(
            shoulder,
            elbow,
            wrist
        )

        visible = (
            shoulder_vis >= SHOULDER_PRESS["min_visibility"]
            and elbow_vis >= SHOULDER_PRESS["min_visibility"]
            and wrist_vis >= SHOULDER_PRESS["min_visibility"]
        )

        if visible:

            if elbow_angle > SHOULDER_PRESS["up_threshold"]:
                stage = "up"

            elif (
                elbow_angle < SHOULDER_PRESS["down_threshold"]
                and stage == "up"
                and now - last_rep_time > REP_COOLDOWN
            ):
                count += 1
                stage = "down"
                last_rep_time = now
                speak(f"Shoulder Press {count}")

        if elbow_angle >= 160:
            extension_status = "FULL EXTENSION"
        elif elbow_angle >= 130:
            extension_status = "NEARLY EXTENDED"
        elif elbow_angle >= 90:
            extension_status = "PRESSING"
        else:
            extension_status = "START POSITION"

        back_angle = calculate_angle(
            shoulder,
            hip,
            knee
        )

        if back_angle >= 160:
            back_arch_status = "Neutral"
        elif back_angle >= 140:
            back_arch_status = "Slight Arch"
        else:
            back_arch_status = "Excessive Arch"

        correct_posture = (
            visible
            and back_angle >= 140
        )

        progress = interp_clip(
            elbow_angle,
            (
                SHOULDER_PRESS["down_threshold"],
                SHOULDER_PRESS["up_threshold"]
            )
        )

        posture_score = interp_clip(
            back_angle,
            (140, 180)
        )
    
    
    elif exercise_name == "Mountain Climbers":

        left_hip = get_joint("hip", "LEFT")
        left_knee = get_joint("knee", "LEFT")
        left_ankle = get_joint("ankle", "LEFT")

        right_hip = get_joint("hip", "RIGHT")
        right_knee = get_joint("knee", "RIGHT")
        right_ankle = get_joint("ankle", "RIGHT")

        left_shoulder = get_joint("shoulder", "LEFT")
        right_shoulder = get_joint("shoulder", "RIGHT")

        visible = (
            left_hip[2] >= MOUNTAIN_CLIMBER["min_visibility"]
            and left_knee[2] >= MOUNTAIN_CLIMBER["min_visibility"]
            and right_hip[2] >= MOUNTAIN_CLIMBER["min_visibility"]
            and right_knee[2] >= MOUNTAIN_CLIMBER["min_visibility"]
        )

        left_leg_length = np.linalg.norm(
            np.array(left_hip[:2]) - np.array(left_ankle[:2])
        )

        right_leg_length = np.linalg.norm(
            np.array(right_hip[:2]) - np.array(right_ankle[:2])
        )

        left_knee_ratio = (
            np.linalg.norm(
                np.array(left_knee[:2]) - np.array(left_shoulder[:2])
            ) / max(left_leg_length, 1e-6)
        )

        right_knee_ratio = (
            np.linalg.norm(
                np.array(right_knee[:2]) - np.array(right_shoulder[:2])
            ) / max(right_leg_length, 1e-6)
        )

        left_knee_up = left_knee_ratio < MOUNTAIN_CLIMBER["knee_threshold"]
        right_knee_up = right_knee_ratio < MOUNTAIN_CLIMBER["knee_threshold"]

        if visible:

            if left_knee_up and stage != "left":
                stage = "left"

                if now - last_rep_time > REP_COOLDOWN:
                    count += 1
                    last_rep_time = now
                    speak(f"Mountain Climber {count}")

            elif right_knee_up and stage != "right":
                stage = "right"

                if now - last_rep_time > REP_COOLDOWN:
                    count += 1
                    last_rep_time = now
                    speak(f"Mountain Climber {count}")

        progress = 100 if (left_knee_up or right_knee_up) else 0

        correct_posture = (
            visible
            and not (left_knee_up and right_knee_up)
        )

        posture_score = 100 if correct_posture else 70
    elif exercise_name == "Glute Bridge":

        hip_angle = calculate_angle(
            shoulder,
            hip,
            knee
        )

        visible = (
            shoulder_vis >= GLUTE_BRIDGE["min_visibility"]
            and hip_vis >= GLUTE_BRIDGE["min_visibility"]
            and knee_vis >= GLUTE_BRIDGE["min_visibility"]
        )

        if visible:

            if hip_angle > GLUTE_BRIDGE["up_threshold"]:
                stage = "up"

            elif (
                hip_angle < GLUTE_BRIDGE["down_threshold"]
                and stage == "up"
                and now - last_rep_time > REP_COOLDOWN
            ):
                count += 1
                stage = "down"
                last_rep_time = now
                speak(f"Glute Bridge {count}")

        progress = interp_clip(
            hip_angle,
            (
                GLUTE_BRIDGE["down_threshold"],
                GLUTE_BRIDGE["up_threshold"]
            )
        )

        correct_posture = (
            visible
            and hip_angle > 140
        )

        posture_score = interp_clip(
            hip_angle,
            (100, 180)
        )
    # ------------------ Push-up ------------------
    elif exercise_name == 'Push-ups':

        # Elbow angle:
        # < 90 degrees  -> DOWN
        # > 160 degrees -> UP
        #
        # One repetition:
        # DOWN -> UP

        key_landmarks_visible = (
            shoulder_vis >= PUSHUP['min_visibility']
            and elbow_vis >= PUSHUP['min_visibility']
            and wrist_vis >= PUSHUP['min_visibility']
            and hip_vis >= PUSHUP['min_visibility']
            and ankle_vis >= PUSHUP['min_visibility']
        )

        elbow_angle = calculate_angle(
            shoulder,
            elbow,
            wrist
        )

        body_angle = calculate_angle(
            shoulder,
            hip,
            ankle
        )

        expected_hip_y = (
            shoulder_y + ankle_y
        ) / 2

        hip_deviation = (
            hip_y - expected_hip_y
        )

        if body_angle > 160:
            body_alignment = "Straight"
        elif body_angle > 140:
            body_alignment = "Slight Bend"
        else:
            body_alignment = "Poor Form"

        if abs(hip_deviation) <= PUSHUP['hip_sag_tolerance']:
            hip_status = "LEVEL"
        elif hip_deviation > PUSHUP['hip_sag_tolerance']:
            hip_status = "SAGGING"
        else:
            hip_status = "PIKED UP"

        posture_score = interp_clip(
            body_angle,
            (140, 180)
        )

        correct_posture = (
            key_landmarks_visible
            and body_alignment != "Poor Form"
            and hip_status == "LEVEL"
        )

        progress = interp_clip(
            elbow_angle,
            (
                PUSHUP['down_threshold'],
                PUSHUP['up_threshold']
            )
        )

        buf = state.setdefault(
            'angle_buf',
            deque(maxlen=SMOOTHING_WINDOW)
        )
        buf.append(progress)
        smooth_progress = float(np.mean(buf))

        if correct_posture:
            if posture_ok_since is None:
                posture_ok_since = now
        else:
            posture_ok_since = None

        posture_ready = (
            posture_ok_since is not None
            and
            (now - posture_ok_since) > POSTURE_HOLD_SEC
        )

        if posture_ready:

            if elbow_angle < PUSHUP['down_threshold']:
                stage = 'down'

            if (
                elbow_angle > PUSHUP['up_threshold']
                and
                stage == 'down'
                and
                (now - last_rep_time) > REP_COOLDOWN
            ):
                count += 1
                last_rep_time = now
                stage = 'up'
                speak(f"Push-up {count}")

        progress = smooth_progress

    elif exercise_name == 'Plank':
        # For plank we check a reference straightness shoulder-hip-ankle
        ref_point = ankle
        ref_angle = calculate_angle(shoulder, hip, ref_point)
        posture_score = interp_clip(ref_angle, PLANK['ref_angle_range'])
        correct_posture = posture_score > PLANK['back_ok']
        if correct_posture:
            hold_start = hold_start or now
            elapsed = now - hold_start
            progress = np.clip((elapsed / PLANK['hold_seconds']) * 100, 0, 100)
        else:
            hold_start = 0
            progress = 0

    # update state
    state['stage'] = stage
    state['count'] = count
    state['last_rep_time'] = last_rep_time
    state['hold_start'] = hold_start
    state['posture_ok_since'] = posture_ok_since

    return state, correct_posture, float(progress), float(posture_score)

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
