import numpy as np
from typing import List, Dict, Tuple
import cv2
from ultralytics import YOLO
import random
import time

class TargetDetector:
    def __init__(self):
        # Инициализация YOLO модели
        self.model = YOLO('yolov8n.pt')
        self.confidence_threshold = 0.5
        self.target_classes = ['person', 'car', 'truck', 'tank']  # Настраиваемый список целей
        self.tracked_targets: Dict[int, Dict] = {}
        self.next_target_id = 0
        
        # Параметры для синтетической генерации целей
        self.synthetic_mode = True  # Режим синтетических данных
        self.synthetic_targets: Dict[int, Dict] = {}
        self.target_spawn_rate = 0.02  # Вероятность появления новой цели на каждой итерации
        self.target_move_speed = 2.0  # м/с
        self.max_synthetic_targets = 20
        self.world_bounds = {
            'x_min': -3000.0, 'x_max': 3000.0,
            'y_min': -3000.0, 'y_max': 3000.0
        }
        self.last_update_time = time.time()

    def detect_targets(self, frame: np.ndarray, drone_position: np.ndarray, camera_params: Dict) -> List[Dict]:
        """Обнаружение целей на кадре"""
        
        if self.synthetic_mode:
            # Обновление синтетических целей
            self._update_synthetic_targets()
            return self._detect_synthetic_targets(drone_position, camera_params)
        
        # Реальная детекция с YOLO если не синтетические данные
        # Запуск детекции YOLO
        results = self.model(frame)[0]
        detected_targets = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            
            if conf < self.confidence_threshold:
                continue

            class_name = self.model.names[int(cls)]
            if class_name not in self.target_classes:
                continue

            # Расчет центра цели в пикселях
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Оценка расстояния до цели
            target_position = self._estimate_3d_position(
                center_x, center_y,
                frame.shape[1], frame.shape[0],
                drone_position,
                camera_params
            )

            target_info = {
                'position': target_position,
                'confidence': conf,
                'class': class_name,
                'bbox': (x1, y1, x2, y2)
            }
            detected_targets.append(target_info)

        return detected_targets

    def _update_synthetic_targets(self):
        """Обновление синтетических целей"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Создание новых целей с определенной вероятностью
        if random.random() < self.target_spawn_rate and len(self.synthetic_targets) < self.max_synthetic_targets:
            self._spawn_synthetic_target()
        
        # Обновление позиций существующих целей
        for target_id, target in self.synthetic_targets.items():
            # Случайное движение
            # Если цель еще не имеет направления движения, создаем его
            if 'direction' not in target:
                target['direction'] = np.array([
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    0  # Движение только по плоскости XY
                ])
                # Нормализуем направление
                if np.linalg.norm(target['direction']) > 0:
                    target['direction'] = target['direction'] / np.linalg.norm(target['direction'])
            
            # Обновление позиции
            target['position'] += target['direction'] * self.target_move_speed * dt
            
            # Проверка границ мира
            if target['position'][0] < self.world_bounds['x_min'] or target['position'][0] > self.world_bounds['x_max']:
                target['direction'][0] *= -1  # Отражение от границы X
                
            if target['position'][1] < self.world_bounds['y_min'] or target['position'][1] > self.world_bounds['y_max']:
                target['direction'][1] *= -1  # Отражение от границы Y
                
            # Случайное изменение направления с небольшой вероятностью
            if random.random() < 0.01:
                angle_change = random.uniform(-0.3, 0.3)  # Максимальное изменение угла +-0.3 радиан
                
                # Вращение вектора направления
                cos_angle = np.cos(angle_change)
                sin_angle = np.sin(angle_change)
                
                new_x = target['direction'][0] * cos_angle - target['direction'][1] * sin_angle
                new_y = target['direction'][0] * sin_angle + target['direction'][1] * cos_angle
                
                target['direction'][0] = new_x
                target['direction'][1] = new_y
                
                # Нормализация вектора
                if np.linalg.norm(target['direction']) > 0:
                    target['direction'] = target['direction'] / np.linalg.norm(target['direction'])

    def _spawn_synthetic_target(self):
        """Создание новой синтетической цели"""
        # Случайная позиция в пределах мира
        position = np.array([
            random.uniform(self.world_bounds['x_min'], self.world_bounds['x_max']),
            random.uniform(self.world_bounds['y_min'], self.world_bounds['y_max']),
            0.0  # Высота цели на земле
        ])
        
        # Случайный класс цели
        target_class = random.choice(self.target_classes)
        
        # Случайный размер цели в зависимости от класса
        size = {
            'person': random.uniform(0.5, 2.0),
            'car': random.uniform(2.0, 4.0),
            'truck': random.uniform(4.0, 8.0),
            'tank': random.uniform(5.0, 10.0)
        }.get(target_class, 1.0)
        
        # Создание новой цели
        target_id = self.next_target_id
        self.next_target_id += 1
        
        self.synthetic_targets[target_id] = {
            'id': target_id,
            'position': position,
            'class': target_class,
            'size': size,
            'confidence': random.uniform(0.7, 0.95),
            'direction': None  # Будет установлено при первом обновлении
        }
        
        print(f"New synthetic target #{target_id} spawned: {target_class} at position {position}")

    def _detect_synthetic_targets(self, drone_position: np.ndarray, camera_params: Dict) -> List[Dict]:
        """Обнаружение синтетических целей в поле зрения дрона"""
        detected_targets = []
        
        # Поле зрения камеры дрона
        fov_h = camera_params['fov_horizontal']
        fov_v = camera_params['fov_vertical']
        max_detection_range = 1000.0  # Максимальный радиус обнаружения в метрах
        
        # Проверяем каждую синтетическую цель
        for target_id, target in self.synthetic_targets.items():
            # Вектор от дрона к цели
            direction_to_target = target['position'] - drone_position
            distance = np.linalg.norm(direction_to_target)
            
            # Если цель слишком далеко, пропускаем
            if distance > max_detection_range:
                continue
                
            # Нормализация вектора направления
            if distance > 0:
                direction_to_target = direction_to_target / distance
            
            # Вектор направления камеры дрона (предполагаем, что смотрит вперед и вниз)
            # По умолчанию дрон смотрит в направлении +Y и немного вниз
            drone_forward = np.array([0, 1, -0.3])
            drone_forward = drone_forward / np.linalg.norm(drone_forward)
            
            # Вычисление угла между направлением камеры и направлением к цели
            cos_angle = np.dot(drone_forward, direction_to_target)
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            
            # Проверка, находится ли цель в поле зрения камеры
            if angle_deg <= max(fov_h, fov_v) / 2:
                # Вычисление уверенности в зависимости от расстояния
                distance_factor = 1.0 - distance / max_detection_range
                detection_confidence = target['confidence'] * distance_factor
                
                # Проверка порога уверенности
                if detection_confidence >= self.confidence_threshold:
                    # Создание bbox для отображения на кадре
                    # В реальной системе это будет результат работы YOLO
                    image_width, image_height = camera_params['resolution']
                    
                    # Проекция 3D позиции цели на плоскость кадра
                    pixel_coords = self._project_3d_to_2d(
                        target['position'],
                        drone_position,
                        camera_params
                    )
                    
                    # Размер bbox на кадре зависит от расстояния и размера цели
                    apparent_size = target['size'] * 50 / max(distance, 1.0)  # Пикселей
                    
                    x1 = max(0, pixel_coords[0] - apparent_size / 2)
                    y1 = max(0, pixel_coords[1] - apparent_size)
                    x2 = min(image_width, pixel_coords[0] + apparent_size / 2)
                    y2 = min(image_height, pixel_coords[1])
                    
                    target_info = {
                        'id': target_id,
                        'position': target['position'],
                        'confidence': detection_confidence,
                        'class': target['class'],
                        'bbox': (x1, y1, x2, y2),
                        'distance': distance
                    }
                    
                    detected_targets.append(target_info)
        
        return detected_targets

    def _project_3d_to_2d(self, target_position: np.ndarray, drone_position: np.ndarray, camera_params: Dict) -> Tuple[float, float]:
        """Проекция 3D координат цели на 2D плоскость кадра"""
        # Параметры камеры
        image_width, image_height = camera_params['resolution']
        fov_h = camera_params['fov_horizontal']
        fov_v = camera_params['fov_vertical']
        
        # Вектор от дрона к цели
        direction = target_position - drone_position
        
        # Направления камеры (предполагаем, что камера смотрит вперед в +Y и немного вниз)
        cam_forward = np.array([0, 1, -0.3])
        cam_forward = cam_forward / np.linalg.norm(cam_forward)
        
        cam_right = np.array([1, 0, 0])  # Вправо от камеры по оси X
        cam_up = np.cross(cam_right, cam_forward)
        cam_up = cam_up / np.linalg.norm(cam_up)
        
        # Проекция направления к цели на плоскость камеры
        right_proj = np.dot(direction, cam_right)
        up_proj = np.dot(direction, cam_up)
        forward_proj = np.dot(direction, cam_forward)
        
        # Конвертация в углы относительно центра камеры
        angle_h = np.arctan2(right_proj, forward_proj)
        angle_v = np.arctan2(up_proj, forward_proj)
        
        # Нормализация в диапазон [-1, 1] для углов в пределах FOV
        norm_h = angle_h / np.radians(fov_h / 2)
        norm_v = angle_v / np.radians(fov_v / 2)
        
        # Преобразование в пиксельные координаты
        pixel_x = (norm_h + 1) / 2 * image_width
        pixel_y = (1 - (norm_v + 1) / 2) * image_height  # Инвертируем Y (в изображении y=0 сверху)
        
        return (pixel_x, pixel_y)

    def _estimate_3d_position(self, 
                            pixel_x: float, 
                            pixel_y: float, 
                            image_width: int, 
                            image_height: int,
                            drone_position: np.ndarray,
                            camera_params: Dict) -> np.ndarray:
        """Оценка 3D позиции цели"""
        # Получение параметров камеры
        fov_h = camera_params['fov_horizontal']
        fov_v = camera_params['fov_vertical']
        altitude = drone_position[2]

        # Преобразование пиксельных координат в углы
        angle_h = ((pixel_x / image_width) - 0.5) * fov_h
        angle_v = ((pixel_y / image_height) - 0.5) * fov_v

        # Расчет расстояния до цели
        ground_distance = altitude / np.tan(np.radians(90 - angle_v))
        
        # Расчет X и Y координат
        x = drone_position[0] + ground_distance * np.sin(np.radians(angle_h))
        y = drone_position[1] + ground_distance * np.cos(np.radians(angle_h))
        z = 0  # Предполагаем, что цель находится на земле

        return np.array([x, y, z])

    def track_targets(self, detected_targets: List[Dict]) -> Dict[int, Dict]:
        """Отслеживание целей между кадрами"""
        current_targets = {}

        for target in detected_targets:
            if 'id' in target:
                # Для синтетических целей ID уже назначен
                target_id = target['id']
            else:
                # Для реальных целей назначаем ID через треккинг
                target_id = self._assign_target_id(target)
                target['id'] = target_id
                
            current_targets[target_id] = target

        # Обновление списка отслеживаемых целей
        self.tracked_targets = current_targets
        return self.tracked_targets

    def _assign_target_id(self, new_target: Dict) -> int:
        """Назначение ID цели"""
        # Поиск соответствия среди существующих целей
        min_distance = float('inf')
        matched_id = None

        for target_id, existing_target in self.tracked_targets.items():
            distance = np.linalg.norm(
                new_target['position'] - existing_target['position']
            )
            if distance < min_distance and distance < 5.0:  # 5 метров threshold
                min_distance = distance
                matched_id = target_id

        if matched_id is not None:
            return matched_id
        else:
            # Создание новой цели
            new_id = self.next_target_id
            self.next_target_id += 1
            return new_id

    def draw_targets(self, frame: np.ndarray, targets: Dict[int, Dict]) -> np.ndarray:
        """Отрисовка обнаруженных целей на кадре"""
        for target_id, target in targets.items():
            if 'bbox' not in target:
                continue
                
            x1, y1, x2, y2 = target['bbox']
            
            # Отрисовка bounding box
            cv2.rectangle(frame, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Цвет в зависимости от класса
            color = {
                'person': (0, 255, 0),    # Зеленый
                'car': (255, 255, 0),     # Желтый
                'truck': (255, 128, 0),   # Оранжевый
                'tank': (0, 0, 255)       # Красный
            }.get(target['class'], (0, 255, 0))
            
            # Отрисовка информации о цели
            label = f"ID: {target_id} | {target['class']} | Conf: {target['confidence']:.2f}"
            if 'distance' in target:
                label += f" | Dist: {target['distance']:.1f}m"
                
            cv2.putText(frame, label, 
                       (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)

        return frame

    def get_target_priority(self, target: Dict) -> float:
        """Расчет приоритета цели"""
        priority = 0.0
        
        # Приоритет на основе класса цели
        class_priority = {
            'tank': 1.0,
            'truck': 0.8,
            'car': 0.6,
            'person': 0.4
        }
        priority += class_priority.get(target['class'], 0.3)
        
        # Приоритет на основе уверенности детекции
        priority += target['confidence'] * 0.3
        
        # Приоритет на основе расстояния (ближе = выше приоритет)
        if 'distance' in target:
            max_distance = 1000.0  # Максимальная дальность обнаружения
            distance_factor = 1.0 - min(target['distance'], max_distance) / max_distance
            priority += distance_factor * 0.2
        
        return priority
        
    def get_synthetic_targets_for_visualization(self) -> List[Dict]:
        """Получение всех синтетических целей для визуализации"""
        return [
            {
                'id': target_id,
                'position': target['position'],
                'class': target['class']
            }
            for target_id, target in self.synthetic_targets.items()
        ] 