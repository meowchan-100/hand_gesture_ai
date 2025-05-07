import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_finger_status(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 Fingers
    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def identify_gesture(finger_status):
    if finger_status == [0, 1, 1, 0, 0]:
        return "Peace ✌️"
    elif finger_status == [0, 1, 0, 0, 0]:
        return "Point ☝️"
    elif finger_status == [0, 0, 0, 0, 0]:
        return "Fist 👊"
    elif finger_status == [1, 1, 1, 1, 1]:
        return "Open Palm 🖐️"
    elif finger_status == [1, 0, 0, 0, 0]:
        return "Thumbs Up 👍"
    else:
        return "Unknown"

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_status = get_finger_status(hand_landmarks)
                gesture = identify_gesture(finger_status)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        cv2.imshow("Hand Gesture AI", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()