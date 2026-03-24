import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import pyttsx3
import time

# ------------------ Speech Engine ------------------
engine = pyttsx3.init()
engine.setProperty('rate', 170)

# ------------------ BMI-based Recommendation ------------------
def recommend_exercise(bmi):
    if bmi < 18.5:
        return ["Push-ups", "Plank", "Bicep Curl"]
    elif bmi < 25:
        return ["Squats", "Lunges", "Plank", "Bicep Curl"]
    else:
        return ["Walking", "Stretching", "Yoga", "Plank"]

# ------------------ Angle Calculation ------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ------------------ Exercise Logic (returns updated state) ------------------
def exercise_logic(exercise_name, landmarks, mp_pose_module, prev_stage, count, last_rep_time):
    """
    returns: (current_stage, count, correct_posture, progress_percent, last_rep_time)
    progress_percent is clamped 0..100
    """
    current_stage = prev_stage
    correct_posture = False
    progress = 0.0

    def get_point(landmark_enum):
        lm = landmarks[landmark_enum.value]
        return [lm.x, lm.y]

    try:
        shoulder = get_point(mp_pose_module.PoseLandmark.LEFT_SHOULDER)
        elbow = get_point(mp_pose_module.PoseLandmark.LEFT_ELBOW)
        wrist = get_point(mp_pose_module.PoseLandmark.LEFT_WRIST)
        hip = get_point(mp_pose_module.PoseLandmark.LEFT_HIP)
        knee = get_point(mp_pose_module.PoseLandmark.LEFT_KNEE)
        ankle = get_point(mp_pose_module.PoseLandmark.LEFT_ANKLE)
    except Exception:
        # missing landmarks -> return unchanged
        return current_stage, count, False, 0.0, last_rep_time

    now = time.time()

    # BICEP CURL: progress mapped from elbow angle (45 -> 100%, 150 -> 0%)
    if exercise_name == "Bicep Curl":
        back_angle = calculate_angle(shoulder, hip, knee)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        correct_posture = 160 < back_angle < 180
        progress = np.interp(elbow_angle, (150, 45), (0, 100))  # inverse mapping
        progress = float(np.clip(progress, 0, 100))

        # use progress threshold + stage to count reliably
        if correct_posture:
            if progress < 10:
                current_stage = "down"   # arm extended
            elif progress > 95 and current_stage == "down":
                if now - last_rep_time > 0.9:
                    count += 1
                    last_rep_time = now
                    current_stage = "up"
                    engine.say(f"Repetition {count}")
                    engine.runAndWait()

    # SQUATS: progress mapped from knee_angle (160 -> 0, 70 -> 100)
    elif exercise_name == "Squats":
        hip_angle = calculate_angle(shoulder, hip, knee)
        knee_angle = calculate_angle(hip, knee, ankle)
        correct_posture = 150 < hip_angle < 200  # looser for hips
        progress = np.interp(knee_angle, (160, 70), (0, 100))
        progress = float(np.clip(progress, 0, 100))

        if correct_posture:
            if progress < 10:
                current_stage = "up"
            elif progress > 95 and current_stage == "up":
                if now - last_rep_time > 0.9:
                    count += 1
                    last_rep_time = now
                    current_stage = "down"
                    engine.say(f"Squat {count}")
                    engine.runAndWait()

    # PLANK: progress maps how straight body is; no rep counting (use hold-based)
    elif exercise_name == "Plank":
        body_angle = calculate_angle(shoulder, hip, ankle)
        correct_posture = 160 < body_angle < 180
        progress = np.interp(body_angle, (140, 180), (0, 100))
        progress = float(np.clip(progress, 0, 100))

    # YOGA: generic posture progress (can be refined per-pose)
    elif exercise_name == "Yoga":
        hip_angle = calculate_angle(shoulder, hip, knee)
        correct_posture = 150 < hip_angle < 200
        progress = np.interp(hip_angle, (140, 180), (0, 100))
        progress = float(np.clip(progress, 0, 100))

    return current_stage, count, correct_posture, progress, last_rep_time

# ------------------ Camera Tracking ------------------
def run_camera_and_track(exercise_name, height, weight):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils

    # Try CAP_DSHOW on Windows, fallback otherwise
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(0)

    time.sleep(0.6)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        engine.say("Could not open camera.")
        engine.runAndWait()
        return

    engine.say(f"Starting {exercise_name}. Get ready!")
    engine.runAndWait()

    count = 0
    stage = None
    last_rep_time = 0.0
    last_posture_state = None

    window_name = "AI Fitness Assistant"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            correct_posture = False
            progress = 0.0

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                stage, count, correct_posture, progress, last_rep_time = exercise_logic(
                    exercise_name, landmarks, mp_pose, stage, count, last_rep_time
                )

                # voice feedback on posture transitions
                if last_posture_state is None:
                    # first frame set
                    last_posture_state = correct_posture
                elif correct_posture != last_posture_state:
                    if correct_posture:
                        engine.say("Good posture. Keep it up.")
                    else:
                        engine.say("Adjust your posture.")
                    engine.runAndWait()
                    last_posture_state = correct_posture

                # draw landmarks
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # draw vertical progress bar (left)
            x1, x2 = 50, 80
            y1, y2 = 150, 400
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
            filled_y = int(y2 - (progress / 100.0) * (y2 - y1))
            fill_color = (0, 255, 0) if correct_posture else (0, 140, 255)  # green if correct else orange
            cv2.rectangle(frame, (x1+2, filled_y), (x2-2, y2-2), fill_color, -1)
            cv2.putText(frame, f"{int(progress)}%", (x1 - 2, y2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # overlay info
            bmi = (weight / (height ** 2)) if height > 0 else 0.0
            cv2.putText(frame, f"BMI: {bmi:.1f}", (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Exercise: {exercise_name}", (110, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Count: {count}", (110, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            posture_text = "✅ Correct" if correct_posture else "❌ Adjust"
            posture_color = (0, 255, 0) if correct_posture else (0, 0, 255)
            cv2.putText(frame, f"Posture: {posture_text}", (110, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, posture_color, 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print("Error during tracking:", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.say("Workout complete. Great job!")
        engine.runAndWait()

# ------------------ GUI Logic ------------------
def start_tracking():
    try:
        height = float(height_entry.get())
        weight = float(weight_entry.get())
        if height <= 0 or weight <= 0:
            result_label.config(text="❌ Enter positive values!")
            return
        bmi = weight / (height ** 2)
        exercises = recommend_exercise(bmi)
        result_label.config(text=f"BMI: {bmi:.2f}\nRecommended: {', '.join(exercises)}")
        selected_exercise = exercise_var.get()
        threading.Thread(target=run_camera_and_track, args=(selected_exercise, height, weight), daemon=True).start()
    except ValueError:
        result_label.config(text="❌ Enter valid numbers!")

# ------------------ Tkinter GUI ------------------
root = tk.Tk()
root.title("AI Fitness Assistant")
root.geometry("460x480")
root.configure(bg="#0f1720")

tk.Label(root, text="AI Fitness Assistant", fg="white", bg="#0f1720", font=("Helvetica", 18, "bold")).pack(pady=12)
tk.Label(root, text="Height (m):", bg="#0f1720", fg="white").pack(anchor="w", padx=20)
height_entry = tk.Entry(root)
height_entry.pack(fill="x", padx=20)

tk.Label(root, text="Weight (kg):", bg="#0f1720", fg="white").pack(anchor="w", padx=20, pady=(8, 0))
weight_entry = tk.Entry(root)
weight_entry.pack(fill="x", padx=20)

tk.Label(root, text="Select Exercise:", bg="#0f1720", fg="white").pack(anchor="w", padx=20, pady=(8, 0))
exercise_var = tk.StringVar()
exercise_dropdown = ttk.Combobox(root, textvariable=exercise_var, state="readonly")
exercise_dropdown['values'] = ("Bicep Curl", "Squats", "Plank", "Yoga")
exercise_dropdown.current(0)
exercise_dropdown.pack(fill="x", padx=20, pady=6)

tk.Button(root, text="Start Exercise", command=start_tracking, bg="#00a3e0", fg="white", font=("Arial", 11, "bold")).pack(pady=12)
result_label = tk.Label(root, text="", bg="#0f1720", fg="lightgreen", font=("Arial", 11))
result_label.pack(pady=6)
tk.Label(root, text="Tip: Press 'q' or close the window to stop.", bg="#0f1720", fg="#9fbddc").pack(pady=10)

root.mainloop()
