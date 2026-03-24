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
        if exercise_name == 'Bicep Curl':
            chosen_side = 'LEFT' if left_wrist_vis >= right_wrist_vis else 'RIGHT'
        elif exercise_name == 'Squats':
            chosen_side = 'LEFT' if left_knee_vis >= right_knee_vis else 'RIGHT'
        else:
            chosen_side = 'LEFT' if left_wrist_vis >= right_wrist_vis else 'RIGHT'
    else:
        chosen_side = side

    # fetch joints for chosen_side
    shoulder_x, shoulder_y, _ = get_joint('shoulder', chosen_side)
    elbow_x, elbow_y, _ = get_joint('elbow', chosen_side)
    wrist_x, wrist_y, _ = get_joint('wrist', chosen_side)
    hip_x, hip_y, _ = get_joint('hip', chosen_side)
    knee_x, knee_y, _ = get_joint('knee', chosen_side)
    ankle_x, ankle_y, _ = get_joint('ankle', chosen_side)

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

    elif exercise_name in ('Plank', 'Yoga'):
        # For plank/yoga we check a reference straightness (shoulder-hip-ankle or knee)
        ref_point = ankle if exercise_name == 'Plank' else knee
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
        return ["Push-ups", "Plank", "Bicep Curl"]
    elif bmi < 25:
        return ["Squats", "Lunges", "Plank", "Bicep Curl"]
    else:
        return ["Walking", "Stretching", "Yoga", "Plank"]

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
                                 values=("Bicep Curl", "Squats", "Plank", "Yoga"))
exercise_dropdown.pack(fill="x", padx=20, pady=6)

tk.Button(root, text="Start Exercise", command=start_tracking,
          bg="#00a3e0", fg="white", font=("Arial", 11, "bold")).pack(pady=12)

result_label = tk.Label(root, text="", bg="#0f1720", fg="lightgreen", font=("Arial", 11))
result_label.pack(pady=6)

tk.Label(root, text="Tip: Press 'q' or close the window to stop.",
         bg="#0f1720", fg="#9fbddc").pack(pady=10)

root.mainloop()
