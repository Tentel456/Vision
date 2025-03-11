import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

class FPVVisionSystem:
    def __init__(self):
        # Initialize the camera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Camera parameters
        self.focal_length = 800
        self.sensor_height = 480
        self.real_height = 55.0
        
        # Initialize tracking history
        self.track_history = {}
        
        # Image enhancement parameters
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.brightness_threshold = 100  # Средняя яркость, ниже которой применяется улучшение

    def enhance_image(self, frame):
        """
        Улучшает качество изображения в условиях плохого освещения
        """
        # Конвертируем в LAB цветовое пространство
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Проверяем среднюю яркость
        mean_brightness = np.mean(l)
        
        if mean_brightness < self.brightness_threshold:
            # Применяем CLAHE к каналу яркости
            l = self.clahe.apply(l)
            
            # Повышаем яркость
            l = cv2.add(l, 30)
            
            # Повышаем контраст
            l = cv2.convertScaleAbs(l, alpha=1.3, beta=0)
        
        # Объединяем каналы обратно
        lab = cv2.merge((l, a, b))
        
        # Конвертируем обратно в BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Применяем шумоподавление
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced

    def calculate_distance(self, bbox, img_height):
        """
        Calculate distance using the object's apparent size and camera parameters
        """
        _, y, _, h = bbox
        object_height_pixels = h
        distance = (self.real_height * self.focal_length) / object_height_pixels
        return distance

    def process_frame(self, frame):
        # Улучшаем качество изображения
        enhanced_frame = self.enhance_image(frame)
        
        # Run YOLOv8 inference на улучшенном кадре
        results = self.model.track(enhanced_frame, persist=True, verbose=False)[0]
        
        # Process detections
        detected_objects = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.cpu().numpy()
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calculate center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Get confidence and class
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                
                # Calculate distance
                distance = self.calculate_distance((x1, y1, x2-x1, y2-y1), enhanced_frame.shape[0])
                
                # Get tracking ID if available
                track_id = int(box.id[0]) if box.id is not None else None
                
                # Update tracking history
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
                
                # Draw bounding box
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(enhanced_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Draw label with distance
                label = f"{class_name} {confidence:.2f} ID:{track_id}"
                cv2.putText(enhanced_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(enhanced_frame, f"Dist: {distance:.2f}cm",
                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                # Draw tracking history
                if track_id in self.track_history:
                    points = np.array(self.track_history[track_id], dtype=np.int32)
                    cv2.polylines(enhanced_frame, [points], False, (0, 255, 255), 2)
        
        # Добавляем информацию о текущей яркости
        mean_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:,:,0])
        cv2.putText(enhanced_frame, f"Brightness: {mean_brightness:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return enhanced_frame, detected_objects

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