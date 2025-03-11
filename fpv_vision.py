import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
from math import atan2, degrees, pi

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
        self.fov_horizontal = 62.2  # Горизонтальный угол обзора камеры в градусах
        self.fov_vertical = 48.8    # Вертикальный угол обзора камеры в градусах
        
        # Initialize tracking history
        self.track_history = {}
        
        # Image enhancement parameters
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.brightness_threshold = 100
        
        # Стабилизация изображения
        self.prev_frame = None
        self.prev_pts = None
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def stabilize_frame(self, frame):
        """
        Стабилизация изображения с помощью оптического потока
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, 
                                                   qualityLevel=0.01, minDistance=30, 
                                                   blockSize=3)
            return frame

        # Вычисляем оптический поток
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, 
                                                        self.prev_pts, None, 
                                                        **self.lk_params)

        # Выбираем хорошие точки
        good_old = self.prev_pts[status == 1]
        good_new = curr_pts[status == 1]

        if len(good_old) < 10 or len(good_new) < 10:
            self.prev_frame = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, 
                                                   qualityLevel=0.01, minDistance=30, 
                                                   blockSize=3)
            return frame

        # Находим матрицу преобразования
        M, _ = cv2.estimateAffinePartial2D(good_old, good_new)
        
        if M is not None:
            # Применяем преобразование
            h, w = frame.shape[:2]
            stabilized = cv2.warpAffine(frame, M, (w, h))
        else:
            stabilized = frame

        # Обновляем предыдущий кадр и точки
        self.prev_frame = gray
        self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, 
                                               qualityLevel=0.01, minDistance=30, 
                                               blockSize=3)

        return stabilized

    def calculate_3d_position(self, bbox, frame_shape):
        """
        Рассчитывает 3D позицию объекта относительно камеры
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        
        # Центр объекта в пикселях
        center_x = x + w/2
        center_y = y + h/2
        
        # Нормализованные координаты (-1 до 1)
        norm_x = (center_x - frame_w/2) / (frame_w/2)
        norm_y = (center_y - frame_h/2) / (frame_h/2)
        
        # Углы в градусах
        angle_x = norm_x * (self.fov_horizontal/2)
        angle_y = norm_y * (self.fov_vertical/2)
        
        # Расстояние
        distance = self.calculate_distance(bbox, frame_h)
        
        # 3D координаты (в см)
        x = distance * np.tan(np.radians(angle_x))
        y = distance * np.tan(np.radians(angle_y))
        z = distance
        
        return x, y, z

    def enhance_image(self, frame):
        """
        Улучшает качество изображения в условиях плохого освещения
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        mean_brightness = np.mean(l)
        
        if mean_brightness < self.brightness_threshold:
            l = self.clahe.apply(l)
            l = cv2.add(l, 30)
            l = cv2.convertScaleAbs(l, alpha=1.3, beta=0)
        
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
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
        # Стабилизируем кадр
        stabilized_frame = self.stabilize_frame(frame)
        
        # Улучшаем качество изображения
        enhanced_frame = self.enhance_image(stabilized_frame)
        
        # Run YOLOv8 inference
        results = self.model.track(enhanced_frame, persist=True, verbose=False)[0]
        
        # Process detections
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
                
                # Получаем 3D координаты
                x3d, y3d, z3d = self.calculate_3d_position(
                    (x1, y1, x2-x1, y2-y1), 
                    enhanced_frame.shape
                )
                
                track_id = int(box.id[0]) if box.id is not None else None
                
                if track_id is not None:
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    self.track_history[track_id].append((center_x, center_y))
                    self.track_history[track_id] = self.track_history[track_id][-30:]
                
                detected_objects.append({
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'center': (center_x, center_y),
                    'position_3d': (x3d, y3d, z3d),
                    'confidence': confidence,
                    'class': class_name,
                    'track_id': track_id
                })
                
                # Draw bounding box
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(enhanced_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Draw label with 3D position
                label = f"{class_name} {confidence:.2f} ID:{track_id}"
                cv2.putText(enhanced_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(enhanced_frame, f"X:{x3d:.1f} Y:{y3d:.1f} Z:{z3d:.1f}cm",
                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                # Draw tracking history
                if track_id in self.track_history:
                    points = np.array(self.track_history[track_id], dtype=np.int32)
                    cv2.polylines(enhanced_frame, [points], False, (0, 255, 255), 2)
        
        # Добавляем информацию о стабилизации и освещении
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