import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

class FPVVisionSystem:
    def __init__(self):
        
        self.camera = cv2.VideoCapture(0)  
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
       
        self.model = YOLO('yolov8n.pt')  
        
        
        self.focal_length = 800  
        self.sensor_height = 480  
        self.real_height = 55.0  
        
        
        self.track_history = {}

    def calculate_distance(self, bbox, img_height):
        
        _, y, _, h = bbox
        
        
        object_height_pixels = h
        
        
        
        distance = (self.real_height * self.focal_length) / object_height_pixels
        
        return distance

    def process_frame(self, frame):
        
        results = self.model.track(frame, persist=True, verbose=False)[0]
        
        
        detected_objects = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.cpu().numpy()
            
            for box in boxes:
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                
                
                distance = self.calculate_distance((x1, y1, x2-x1, y2-y1), frame.shape[0])
                
                
                track_id = int(box.id[0]) if box.id is not None else None
                
                
                if track_id is not None:
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    self.track_history[track_id].append((center_x, center_y))
                    
                    self.track_history[track_id] = self.track_history[track_id][-30:]
                
                detected_objects.append({
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'center': (center_x, center_y),
                    'distance': distance,
                    'confidence': confidence,
                    'class': class_name,
                    'track_id': track_id
                })
                
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                
                label = f"{class_name} {confidence:.2f} ID:{track_id}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Dist: {distance:.2f}cm",
                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                
                if track_id in self.track_history:
                    points = np.array(self.track_history[track_id], dtype=np.int32)
                    cv2.polylines(frame, [points], False, (0, 255, 255), 2)
        
        return frame, detected_objects

    def run(self):
        try:
            while True:
                
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                
                processed_frame, detected_objects = self.process_frame(frame)
                
                
                cv2.imshow("FPV Vision", processed_frame)
                
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    vision_system = FPVVisionSystem()
    vision_system.run() 