import cv2
import mediapipe as mp
import numpy as np 
import tkinter as tk
from tkinter import ttk
import threading 
import pyttsx3
import time

engine = pyttsx3.init()
engine.setProperty('rate', 170)

def recommend_exercise(bmi):
    if bmi < 18.5:
        return ["Push-ups", "Plank", "Bicep Curl"]
    elif bmi < 25:
        return ["Squats", "Lunges", "Plank", "Bicep Curl"]
    else:
        return ["Walking", "Stretching", "Yoga", "Plank"]


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_side_visibility(landmarks, mp_pose):
    left_visibility = sum([
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    ])
    right_visibility = sum([
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
    ])
    return "LEFT" if left_visibility >= right_visibility else "RIGHT"

def exercise_logic(exercise_name, landmarks, mp_pose, prev_stage, count, last_rep_time, hold_start):
    current_stage = prev_stage
    correct_posture = False
    progress = 0.0
    posture_score = 0.0
    now = time.time()

    side = get_side_visibility(landmarks, mp_pose)
    if side == "LEFT":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    else:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Movement logic
    if exercise_name == "Bicep Curl":
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        back_angle = calculate_angle(shoulder, hip, knee)
        posture_score = np.clip(np.interp(back_angle, (150, 180), (0, 100)), 0, 100)
        correct_posture = posture_score > 80
        progress = np.clip(np.interp(elbow_angle, (150, 45), (0, 100)), 0, 100)

        if correct_posture:
            if progress < 10:
                current_stage = "down"
            elif progress > 95 and current_stage == "down" and (now - last_rep_time) > 1:
                count += 1
                current_stage = "up"
                last_rep_time = now
                engine.say(f"Repetition {count}")
                engine.runAndWait()

    elif exercise_name == "Squats":
        knee_angle = calculate_angle(hip, knee, ankle)
        back_angle = calculate_angle(shoulder, hip, knee)
        posture_score = np.clip(np.interp(back_angle, (150, 180), (0, 100)), 0, 100)
        correct_posture = posture_score > 80
        progress = np.clip(np.interp(knee_angle, (160, 70), (0, 100)), 0, 100)

        if correct_posture:
            if progress < 10:
                current_stage = "up"
            elif progress > 95 and current_stage == "up" and (now - last_rep_time) > 1:
                count += 1
                current_stage = "down"
                last_rep_time = now
                engine.say(f"Squat {count}")
                engine.runAndWait()

    elif exercise_name == "Plank":
        body_angle = calculate_angle(shoulder, hip, ankle)
        posture_score = np.clip(np.interp(body_angle, (140, 180), (0, 100)), 0, 100)
        correct_posture = posture_score > 85
        if correct_posture:
            if hold_start == 0:
                hold_start = now
            elapsed = int(now - hold_start)
            progress = np.clip(elapsed * 5, 0, 100)
        else:
            hold_start = 0
            progress = 0

    elif exercise_name == "Yoga":
        hip_angle = calculate_angle(shoulder, hip, knee)
        posture_score = np.clip(np.interp(hip_angle, (140, 180), (0, 100)), 0, 100)
        correct_posture = posture_score > 85
        if correct_posture:
            if hold_start == 0:
                hold_start = now
            elapsed = int(now - hold_start)
            progress = np.clip(elapsed * 5, 0, 100)
        else:
            hold_start = 0
            progress = 0

    return current_stage, count, correct_posture, progress, posture_score, last_rep_time, hold_start


def run_camera_and_track(exercise_name, height, weight):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        engine.say("Could not open camera.")
        engine.runAndWait()
        return

    count = 0
    stage = None
    last_rep_time = 0
    hold_start = 0
    last_posture_state = None

    window_name = "AI Fitness Assistant"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    engine.say(f"Starting {exercise_name}. Get ready!")
    engine.runAndWait()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        correct_posture = False
        progress = 0
        posture_score = 0

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            stage, count, correct_posture, progress, posture_score, last_rep_time, hold_start = exercise_logic(
                exercise_name, landmarks, mp_pose, stage, count, last_rep_time, hold_start
            )

            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if last_posture_state != correct_posture:
                if correct_posture:
                    engine.say("Good posture.")
                else:
                    engine.say("Adjust your posture.")
                engine.runAndWait()
                last_posture_state = correct_posture

        # Left Bar: Movement Progress
        cv2.rectangle(frame, (50, 150), (80, 400), (255, 255, 255), 3)
        filled_y1 = int(400 - (progress * 2.5))
        cv2.rectangle(frame, (50, filled_y1), (80, 400), (0, 255, 0) if correct_posture else (0, 0, 255), -1)
        cv2.putText(frame, "Move", (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Right Bar: Posture Accuracy
        cv2.rectangle(frame, (560, 150), (590, 400), (255, 255, 255), 3)
        filled_y2 = int(400 - (posture_score * 2.5))
        cv2.rectangle(frame, (560, filled_y2), (590, 400), (255, 255, 0) if posture_score > 80 else (0, 0, 255), -1)
        cv2.putText(frame, "Form", (550, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Display Info
        bmi = weight / (height ** 2)
        cv2.putText(frame, f"BMI: {bmi:.1f}", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Exercise: {exercise_name}", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Count: {count}", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Posture: {'Good' if correct_posture else 'Bad'}", (100, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if correct_posture else (0,0,255), 2)

        cv2.imshow(window_name, frame)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    engine.say("Workout complete. Great job!")
    engine.runAndWait()

# ------------------ GUI ------------------
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
