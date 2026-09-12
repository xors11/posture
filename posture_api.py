"""
posture_api.py
==============
Thin frame-processing adapter for the FastAPI web backend.

- Sets WEB_MODE = True BEFORE importing posture_corrector so that
  pyttsx3 is never initialised and speak() is a no-op.
- Imports only the reusable, pure-logic pieces from posture_corrector.
- Exposes `process_frame()` which accepts a raw BGR numpy frame and
  a per-session state bundle, runs the existing MediaPipe / exercise_logic
  pipeline, and returns a plain dict ready to be JSON-serialised.
- No cv2.VideoCapture, no cv2.imshow, no Tkinter, no TTS.
"""

import cv2
import numpy as np
import mediapipe as mp
import time

# ---------------------------------------------------------------------------
# Critical: set WEB_MODE before the import so that posture_corrector's
# module-level speak() guard is in place when the file is first executed.
# ---------------------------------------------------------------------------
import posture_corrector as _pc   # noqa: E402 (import after side-effect)
_pc.WEB_MODE = True               # silence TTS for every frame going forward

# Pull in the pure-logic helpers we need directly so call sites are clean.
from posture_corrector import (   # noqa: E402
    LandmarkSmoother,
    exercise_logic,
    recommend_exercise,
)

# ---------------------------------------------------------------------------
# Supported exercises (mirrors the Tkinter dropdown values)
# ---------------------------------------------------------------------------
SUPPORTED_EXERCISES = [
    "Bicep Curl",
    "Squats",
    "Push-ups",
    "Plank",
    "Lunges",
    "Shoulder Press",
    "Glute Bridge",
    "Mountain Climbers",
    "Jumping Jacks",
    "High Knees",
    "Side Lunges",
    "Side Leg Raises",
    "Wall Sit",
    "Standing Knee-to-Elbow",
    "Arm Circles",
]


# ---------------------------------------------------------------------------
# Per-session state factory
# ---------------------------------------------------------------------------

def create_session():
    """
    Create a fresh, isolated session bundle.

    Each WebSocket connection should call this once on connect and store
    the returned dict.  Nothing is shared between sessions.

    Returns
    -------
    dict with keys:
        pose        – MediaPipe Pose instance (call .close() on disconnect)
        smoother    – LandmarkSmoother
        state       – exercise_logic state dict
        mp_pose     – mp.solutions.pose module reference
        exercise    – currently selected exercise name (str)
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return {
        "pose": pose,
        "smoother": LandmarkSmoother(),
        "state": {
            "stage": None,
            "count": 0,
            "last_rep_time": 0,
            "hold_start": 0,
            "posture_ok_since": None,
        },
        "mp_pose": mp_pose,
        "exercise": "Bicep Curl",  # default; overridden by first client message
    }


def close_session(session: dict):
    """Release the MediaPipe Pose instance held by this session."""
    try:
        session["pose"].close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Core frame-processing function
# ---------------------------------------------------------------------------

def process_frame(frame_bgr: np.ndarray, exercise_name: str, session: dict) -> dict:
    """
    Run one frame through the MediaPipe / exercise_logic pipeline.

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR image decoded from the client's JPEG (via cv2.imdecode).
    exercise_name : str
        One of SUPPORTED_EXERCISES.
    session : dict
        The per-session bundle returned by create_session().

    Returns
    -------
    dict
        {
            "correct_posture": bool,
            "progress":        float,   # 0–100
            "posture_score":   float,   # 0–100
            "rep_count":       int,
            "exercise":        str,
            "feedback":        str,
        }
    """
    pose = session["pose"]
    smoother = session["smoother"]
    state = session["state"]
    mp_pose = session["mp_pose"]

    # Update the exercise stored in session so the caller can inspect it.
    session["exercise"] = exercise_name

    # ------------------------------------------------------------------ #
    # 1. Convert BGR → RGB for MediaPipe                                  #
    # ------------------------------------------------------------------ #
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False          # perf hint for MediaPipe
    result = pose.process(rgb)
    rgb.flags.writeable = True

    # ------------------------------------------------------------------ #
    # 2. If MediaPipe found no landmarks at all → return neutral result   #
    # ------------------------------------------------------------------ #
    if not result.pose_landmarks:
        return {
            "correct_posture": False,
            "progress": 0.0,
            "posture_score": 0.0,
            "rep_count": state.get("count", 0),
            "exercise": exercise_name,
            "feedback": "No pose detected – make sure your full body is visible.",
            "landmarks": None,
        }

    # ------------------------------------------------------------------ #
    # 3. Feed landmarks into the smoother                                 #
    # ------------------------------------------------------------------ #
    for i, lm in enumerate(result.pose_landmarks.landmark):
        smoother.add(i, lm.x, lm.y, lm.visibility)

    # ------------------------------------------------------------------ #
    # 4. Run the existing exercise_logic (unchanged from posture_corrector) #
    # ------------------------------------------------------------------ #
    now = time.time()
    state, correct_posture, progress, posture_score = exercise_logic(
        exercise_name,
        result.pose_landmarks.landmark,
        mp_pose,
        smoother,
        state,
        now,
    )

    # Persist updated state back into session
    session["state"] = state

    # ------------------------------------------------------------------ #
    # 5. Build human-readable feedback string                             #
    #                                                                      #
    # FIX: MediaPipe can return pose_landmarks (step 2 above passes) while #
    # exercise_logic *still* decides the required joints aren't visible   #
    # enough for the chosen exercise (see side_and_visibility's           #
    # MIN_VISIBILITY gate). Previously that case fell through to the      #
    # generic per-exercise form hint (e.g. "Keep your back straight and   #
    # elbow close to your torso"), which is actively misleading when the  #
    # real problem is that the user's shoulders/elbows/hips are out of    #
    # frame or occluded — exactly what happened in the reviewed           #
    # recording when the user leaned close to the camera. We now check    #
    # state['low_visibility'] (set inside exercise_logic) and give a      #
    # framing-specific hint instead.                                      #
    # ------------------------------------------------------------------ #
    low_visibility = bool(state.get("low_visibility", False))
    form_issue = state.get("form_issue")  # None | 'back' | 'elbow' | 'both' (Bicep Curl only, for now)
    feedback = _build_feedback(correct_posture, posture_score, exercise_name, low_visibility, form_issue)

    # ------------------------------------------------------------------ #
    # 6. Serialise pose landmarks (normalised 0-1 coords, 33 points)     #
    #                                                                      #
    # FIX (skeleton jitter): this used to serialise the RAW per-frame     #
    # MediaPipe landmarks even though `smoother` (a 5-frame rolling       #
    # average, fed just above in step 3) already existed and was used     #
    # for all the exercise_logic math. The drawn skeleton was therefore   #
    # visually noisier than the numbers driving rep counting/form score. #
    # Now we send the same smoothed coordinates exercise_logic itself     #
    # relies on, so what's drawn matches what's measured and jitters      #
    # far less. Falls back to the raw landmark if the smoother somehow    #
    # has no buffered value yet (first frame of a session).               #
    # ------------------------------------------------------------------ #
    landmarks = []
    for i, lm in enumerate(result.pose_landmarks.landmark):
        sm = smoother.smoothed(i)
        if sm is not None:
            sx, sy, svis = sm
        else:
            sx, sy, svis = lm.x, lm.y, lm.visibility
        landmarks.append({
            "x": round(float(sx), 5),
            "y": round(float(sy), 5),
            "z": round(lm.z, 5),           # depth isn't smoothed/used for drawing math
            "visibility": round(float(svis), 4),
        })

    return {
        "correct_posture": bool(correct_posture),
        "progress": round(float(progress), 2),
        "posture_score": round(float(posture_score), 2),
        "rep_count": int(state.get("count", 0)),
        "exercise": exercise_name,
        "feedback": feedback,
        "landmarks": landmarks,
    }



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_feedback(correct_posture: bool, posture_score: float, exercise: str,
                     low_visibility: bool = False, form_issue: str = None) -> str:
    """Return a short, human-readable posture hint."""
    if low_visibility:
        return "Step back so your shoulders, elbows, and hips are all clearly visible."

    if correct_posture:
        if posture_score >= 90:
            return "Excellent form! Keep it up."
        elif posture_score >= 75:
            return "Good posture. Stay controlled."
        else:
            return "Posture OK. Try to improve your form."
    else:
        # FIX: Bicep Curl now tracks *why* form is bad (state['form_issue'],
        # set in exercise_logic) instead of always showing the combined
        # "back straight and elbow close" hint even when only one of those
        # was actually wrong (e.g. elbow flared/pulled back while the back
        # was fine, or vice versa).
        if exercise == "Bicep Curl" and form_issue:
            bicep_hints = {
                "elbow": "Keep your elbow tucked close to your torso — don't let it drift back or out.",
                "back":  "Keep your back straight — avoid leaning or arching.",
                "both":  "Keep your back straight and your elbow close to your torso.",
            }
            if form_issue in bicep_hints:
                return bicep_hints[form_issue]

        if exercise == "Squats" and form_issue:
            squat_hints = {
                "knee": "Keep your knees tracking over your toes — don't let them cave in or shoot forward.",
                "back": "Keep your back upright — avoid rounding or leaning too far forward.",
                "both": "Keep your back upright and your knees tracking over your toes.",
            }
            if form_issue in squat_hints:
                return squat_hints[form_issue]

        # Give exercise-specific hints for common mistakes
        hints = {
            "Bicep Curl":    "Keep your back straight and elbow close to your torso.",
            "Squats":        "Keep your back upright and knees over your toes.",
            "Push-ups":      "Maintain a straight body line – avoid hip sag or piking.",
            "Plank":         "Align your body from head to heels; don't let your hips drop.",
            "Lunges":        "Keep your torso upright and front knee above the ankle.",
            "Shoulder Press":"Keep your back neutral – avoid excessive arching.",
            "Glute Bridge":  "Press through your heels and drive your hips fully upward.",
            "Mountain Climbers": "Keep hips level – don't let them rise as you drive your knees.",
            "Jumping Jacks": "Bring arms fully overhead and land softly.",
            "High Knees":    "Drive knees up to hip height alternately, stay on the balls of your feet.",
            "Side Lunges":   "Push your hips back and keep the active knee above the foot.",
            "Side Leg Raises": "Keep the leg straight and avoid swinging.",
            "Wall Sit":      "Maintain a 90-degree knee angle with your back flat against the wall.",
            "Standing Knee-to-Elbow": "Bring knee and opposite elbow together at the midline.",
            "Arm Circles":   "Keep arms fully extended at shoulder height.",
        }
        return hints.get(exercise, "Adjust your posture and try again.")