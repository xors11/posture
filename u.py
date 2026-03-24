import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import pyttsx3
import time

engine = pyttsx3.init()

# ------------------ BASIC MEDIAPIPE FEED ------------------
def open_mediapipe_feed():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(1)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Mediapipe Feed", frame)

        if cv2.getWindowProperty("Mediapipe Feed", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------ EXERCISE RECOMMENDATION ------------------
def recommend_exercise(bmi):
    if bmi < 18.5:
        return ["Push-ups", "Plank", "Bicep Curl"]
    elif bmi < 25:
        return ["Squats", "Lunges", "Plank", "Bicep Curl"]
    else:
        return ["Walking", "Stretching", "Yoga", "Bicep Curl"]


# ------------------ HELPER FUNCTION ------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


# ------------------ CAMERA & POSTURE TRACKING ------------------
def run_camera_and_track(exercise_name, height, weight):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils

    count = 0
    stage = None
    last_posture_correct = False
    last_rep_time = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(1)

    if not cap.isOpened():
        print("❌ Error: Could not open video stream.")
        return

    engine.say(f"Starting {exercise_name}. Get ready!")
    engine.runAndWait()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)
            correct_posture = False

            if exercise_name == "Bicep Curl" and result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
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

                back_angle = calculate_angle(shoulder, hip, knee)
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                # ✅ Posture Check
                if 160 < back_angle < 180:
                    correct_posture = True
                    if not last_posture_correct:
                        engine.say("Posture correct. Continue.")
                        engine.runAndWait()
                        last_posture_correct = True
                else:
                    correct_posture = False
                    if last_posture_correct:
                        engine.say("Adjust your posture. Keep your back straight.")
                        engine.runAndWait()
                        last_posture_correct = False

                # ✅ Improved Rep Count Logic
                if correct_posture:
                    if elbow_angle > 150 and stage != "down":
                        stage = "down"
                    elif elbow_angle < 45 and stage == "down":
                        current_time = time.time()
                        if current_time - last_rep_time > 0.8:  # debounce
                            stage = "up"
                            count += 1
                            last_rep_time = current_time
                            print(f"✅ Rep {count}")
                            engine.say(f"Repetition {count}")
                            engine.runAndWait()

                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display Info
            bmi = weight / (height ** 2) if height > 0 else 0
            cv2.putText(frame, f"BMI: {bmi:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Exercise: {exercise_name}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Count: {count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, "Posture: " + ("✅ Correct" if correct_posture else "❌ Adjust"),
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if correct_posture else (0, 0, 255), 2)

            cv2.imshow("Fitness Assistant", frame)

            # ✅ Detect window close or 'q'
            if cv2.getWindowProperty("Fitness Assistant", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.say("Workout complete. Good job!")
        engine.runAndWait()


# ------------------ GUI ------------------
def start_tracking():
    try:
        height = float(height_entry.get())
        weight = float(weight_entry.get())
        bmi = weight / (height ** 2)
        exercises = recommend_exercise(bmi)

        result_label.config(text=f"BMI: {bmi:.2f}\nRecommended: {', '.join(exercises)}")

        selected_exercise = exercise_var.get()
        threading.Thread(target=run_camera_and_track, args=(selected_exercise, height, weight)).start()

    except ValueError:
        result_label.config(text="❌ Enter valid height and weight!")


# ------------------ TKINTER UI ------------------
root = tk.Tk()
root.title("AI Fitness Assistant")
root.geometry("400x400")

tk.Label(root, text="Height (m):").pack()
height_entry = tk.Entry(root)
height_entry.pack()

tk.Label(root, text="Weight (kg):").pack()
weight_entry = tk.Entry(root)
weight_entry.pack()

tk.Label(root, text="Select Exercise:").pack()
exercise_var = tk.StringVar()
exercise_dropdown = ttk.Combobox(root, textvariable=exercise_var)
exercise_dropdown['values'] = ("Bicep Curl", "Squats", "Plank", "Yoga")
exercise_dropdown.current(0)
exercise_dropdown.pack()

tk.Button(root, text="Start Exercise", command=start_tracking).pack(pady=10)
tk.Button(root, text="Open Mediapipe Feed", command=lambda: threading.Thread(target=open_mediapipe_feed).start()).pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
