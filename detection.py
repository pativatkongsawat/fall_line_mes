import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import requests
from tensorflow.keras.models import load_model


model = load_model('lstm_fall_detection.h5')


SELECTED_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
sequence_length = 12
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)


ALERT_API_URL = "http://127.0.0.1:8000/alert_fall"
VIDEO_URL = "http://localhost:8000/videos/video_99.avi" 

def detectPose(image, pose):
    imgHeight, imgWidth, _ = image.shape
    imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    
    landmarks = []
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in SELECTED_LANDMARKS:
                landmarks.append((landmark.x * imgWidth, landmark.y * imgHeight))
    return results, landmarks

def predict_fall(model, sequence):
    sequence = np.array(sequence)
    sequence = sequence.reshape(1, sequence_length, len(SELECTED_LANDMARKS) * 2)
    prediction = model.predict(sequence)
    return prediction[0][0] > 0.5  

def send_fall_alert():
    try:
        data = {"video_url": VIDEO_URL}
        response = requests.post(ALERT_API_URL, json=data)
        print(f"📢 แจ้งเตือนการล้ม: {response.json()}")
    except Exception as e:
        print(f"❌ ไม่สามารถแจ้งเตือนการล้ม: {e}")

def process_video(video_path, model):
    cap = cv.VideoCapture(video_path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    sequence = []
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        results, landmarks = detectPose(frame, pose)
        
        if landmarks:
            flattened_landmarks = [coord for landmark in landmarks for coord in landmark]
            sequence.append(flattened_landmarks)
            
            if len(sequence) >= sequence_length:
                is_fall = predict_fall(model, sequence[-sequence_length:])
                if is_fall:
                    cv.putText(frame, "FALL", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    send_fall_alert() 
                else:
                    cv.putText(frame, "ADL", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv.imshow("Fall Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

# เรียกใช้งาน
video_path = "video_test/adl/video_339_flip.avi"
process_video(video_path, model)
