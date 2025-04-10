import cv2
import mediapipe as mp
import pyautogui
import time
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
drawing_utils = mp.solutions.drawing_utils

BLINK_THRESHOLD = 0.25
blink_detected = False
last_jump_time = 0
cooldown = 1

LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 380, 373]

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_landmark_coords(landmarks, indices, w, h):
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

def calculate_ear(eye_points):
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_eye = get_landmark_coords(landmarks, LEFT_EYE_LANDMARKS, w, h)
            right_eye = get_landmark_coords(landmarks, RIGHT_EYE_LANDMARKS, w, h)

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if avg_ear < BLINK_THRESHOLD:
                if not blink_detected and time.time() - last_jump_time > cooldown:
                    pyautogui.press("space")
                    print("BLINK DETECTED - JUMP!")
                    blink_detected = True
                    last_jump_time = time.time()
                    cv2.putText(frame, "Jump!", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                blink_detected = False

            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

    cv2.imshow("Blink Dino Controller 👁️🦖", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()