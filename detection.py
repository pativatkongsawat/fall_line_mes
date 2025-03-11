import cv2
import mediapipe as mp
import numpy as np
import requests
from collections import deque
from tensorflow.keras.models import load_model


model = load_model("lstm_fall_detection.h5")


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

SELECTED_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)
fall_detected = False
API_URL = "http://127.0.0.1:8000/alert_fall"

def extract_landmarks(landmarks):
    """ ดึงค่า x, y ของจุดสำคัญที่เลือก """
    return np.array([[landmarks[i].x, landmarks[i].y] for i in SELECTED_LANDMARKS]).flatten()

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

       
        landmarks_array = extract_landmarks(results.pose_landmarks.landmark)
        sequence.append(landmarks_array)

      
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(np.array(sequence), axis=0)  
            prediction = model.predict(input_data)[0]  

            if prediction[0] > 0.5:  
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
