import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
gesture_state = None
last_action_time = time.time()

def is_hand_up(landmarks):
    wrist_y = landmarks[0].y
    index_y = landmarks[8].y
    return index_y < wrist_y

def is_fist(landmarks):
    tips_ids = [8, 12, 16, 20]
    for tip_id in tips_ids:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            return False
    return True

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = handLms.landmark

            if time.time() - last_action_time > 1:
                if is_hand_up(landmarks):
                    pyautogui.press("space")
                    print("Jump!")
                    gesture_state = "jump"
                    last_action_time = time.time()

                elif is_fist(landmarks):
                    pyautogui.keyDown("down")
                    print("Duck!")
                    gesture_state = "duck"
                    last_action_time = time.time()
    else:
        if gesture_state == "duck":
            pyautogui.keyUp("down")
            gesture_state = None

    cv2.putText(img, f"Gesture: {gesture_state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Dino Controller 🦖", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()