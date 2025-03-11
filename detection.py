import cv2
import mediapipe as mp
import requests
from collections import deque

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

SELECTED_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)

fall_detected = False
API_URL = "http://127.0.0.1:8000/alert_fall"  

def detect_fall(landmarks):
    global fall_detected
    if not landmarks:
        return False

    nose_y = landmarks[0].y
    hip_y = (landmarks[23].y + landmarks[24].y) / 2  

    if nose_y > hip_y:  
        return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        if detect_fall(results.pose_landmarks.landmark):
            if not fall_detected:
                print("🚨 ตรวจพบการล้ม! ส่งแจ้งเตือน...")
                try:
                    requests.post(API_URL)
                except Exception as e:
                    print("❌ ไม่สามารถส่งแจ้งเตือนได้:", e)

                fall_detected = True  
        else:
            fall_detected = False  

    cv2.imshow('Fall Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
