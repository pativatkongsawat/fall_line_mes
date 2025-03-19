import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import requests
from tensorflow.keras.models import load_model
from pymongo import MongoClient
from datetime import datetime
from tensorflow.keras.metrics import Accuracy
from collections import deque


model = load_model('model/lstm_fall_detection_aug.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Accuracy()])


SELECTED_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
sequence_length = 12  
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)


mp_drawing = mp.solutions.drawing_utils


ALERT_API_URL = "http://127.0.0.1:8000/alert_fall"
client = MongoClient('mongodb://localhost:27017/')
db = client['fall_detection']
collection = db['videos']


sequence_queue = deque(maxlen=sequence_length)


VIDEO_SAVE_DIR = "recorded_videos"
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)

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

def send_fall_alert(video_url):
   
    try:
        data = {"video_url": video_url}
        response = requests.post(ALERT_API_URL, json=data)
        print(f"📢 แจ้งเตือนการล้ม: {response.json()}")
    except Exception as e:
        print(f"❌ ไม่สามารถแจ้งเตือนการล้ม: {e}")

def save_video_to_mongodb(video_path):
   
    with open(video_path, 'rb') as video_file:
        video_data = video_file.read()
    
    video_document = {
        "timestamp": datetime.now(),
        "video_data": video_data,
        "video_url": f"http://localhost:8000/videos/{os.path.basename(video_path)}"
    }
    
    collection.insert_one(video_document)
    return video_document['video_url']

def record_video(cap, duration=5):
    
    fps = 30
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"fall_{timestamp}.avi"
    video_path = os.path.join(VIDEO_SAVE_DIR, video_filename)
    
    out = cv.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    
    start_time = cv.getTickCount()
    while (cv.getTickCount() - start_time) / cv.getTickFrequency() < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    out.release()
    return video_path

def process_webcam(model):
    
    cap = cv.VideoCapture(1)  
    cap.set(cv.CAP_PROP_FPS, 30)  
    sequence = []
    alert_sent = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        
        results, landmarks = detectPose(frame, pose)
        
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mpPose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
     
        if landmarks:
            flattened_landmarks = [coord for landmark in landmarks for coord in landmark]
            sequence_queue.append(flattened_landmarks)
            
           
            if len(sequence_queue) == sequence_length:
                is_fall = predict_fall(model, sequence_queue)
                if is_fall and not alert_sent:
                    cv.putText(frame, "FALL", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    video_path = record_video(cap)
                    video_url = save_video_to_mongodb(video_path)
                    send_fall_alert(video_url)
                    alert_sent = True
                elif not is_fall:
                    cv.putText(frame, "ADL", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    alert_sent = False  
        
       
        cv.imshow("Fall Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    process_webcam(model)