import numpy as np
import cv2
import threading
import time
import random
from typing import List, Dict, Tuple
import argparse
import os

from swarm_controller import SwarmController, DroneState
from drone import Drone
from target_detection import TargetDetector

class DroneSwarmSystem:
    def __init__(self, num_drones: int = 10):
        self.swarm_controller = SwarmController(num_drones)
        self.target_detector = TargetDetector()
        self.running = False
        self.attack_mode = False
        self.drones = {}  # Локальные объекты дронов для симуляции
        
        # Создание директории для логов
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"swarm_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        
        # Параметры камеры
        self.camera_params = {
            'fov_horizontal': 90.0,
            'fov_vertical': 60.0,
            'resolution': (1280, 720)
        }
        
        # Параметры визуализации
        self.map_size = (1200, 900)  # Размер карты в пикселях
        self.scale_factor = 0.1      # Масштаб для преобразования метров в пиксели
        self.center_offset = np.array([self.map_size[0] // 2, self.map_size[1] // 2])
        
        # Создание окон визуализации
        cv2.namedWindow('Drone Swarm Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drone Swarm Control', self.map_size[0], self.map_size[1])
        
        cv2.namedWindow('Drone Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drone Camera View', self.camera_params['resolution'][0], self.camera_params['resolution'][1])
        
        # Инициализация дронов
        self._initialize_drones()
        
        # Выбранный дрон для просмотра с камеры
        self.selected_drone_id = 0
        
        # Статистика
        self.stats = {
            'mission_start_time': 0,
            'targets_detected': 0,
            'targets_attacked': 0,
            'collisions_avoided': 0,
            'max_distance_traveled': 0
        }

    def _initialize_drones(self):
        """Инициализация объектов дронов для симуляции"""
        for i in range(self.swarm_controller.num_drones):
            # Получаем начальную позицию из контроллера
            initial_position = self.swarm_controller.drones[i]['position']
            
            # Создаем объект дрона
            self.drones[i] = Drone(i, initial_position)

    def start(self):
        """Запуск системы управления роем"""
        self.running = True
        self.stats['mission_start_time'] = time.time()
        
        # Запуск всех дронов
        self.swarm_controller.start_mission()
        
        # Запуск основных потоков
        threading.Thread(target=self._main_loop, daemon=True).start()
        threading.Thread(target=self._visualization_loop, daemon=True).start()
        threading.Thread(target=self._drone_simulation_loop, daemon=True).start()
        
        self._log("Drone swarm system started")
        print("Drone swarm system started")
        self._operator_interface()

    def _main_loop(self):
        """Основной цикл обработки"""
        while self.running:
            # Обновление состояния роя
            swarm_status = self.swarm_controller.get_swarm_status()
            
            # Обработка целей если включен режим атаки
            if self.attack_mode:
                self._process_attack_mode()
            
            # Передача информации о ближайших дронах для избежания столкновений
            self._update_collision_avoidance()
            
            # Обновление статистики
            self._update_statistics()
            
            time.sleep(0.1)  # 10Hz update rate

    def _update_statistics(self):
        """Обновление статистики миссии"""
        # Максимальное пройденное расстояние
        for drone_id, drone in self.drones.items():
            self.stats['max_distance_traveled'] = max(
                self.stats['max_distance_traveled'], 
                drone.distance_traveled
            )

    def _update_collision_avoidance(self):
        """Обновление информации для предотвращения столкновений"""
        drone_positions = [(drone_id, drone.position) for drone_id, drone in self.drones.items()]
        
        for drone_id, drone in self.drones.items():
            # Передаем инфу о других дронах, но не о самом себе
            drone.set_nearby_drones(drone_positions)

    def _process_attack_mode(self):
        """Обработка режима атаки"""
        # Получаем информацию о целях от всех дронов
        all_detected_targets = []
        
        for drone_id, drone in self.drones.items():
            if drone.state == DroneState.RTB:
                continue
                
            # Получаем позицию дрона
            drone_position = drone.position
            
            # Создаем виртуальный кадр с камеры дрона
            frame = self._create_drone_camera_frame(drone_id)
            
            # Запуск детекции целей
            detected_targets = self.target_detector.detect_targets(
                frame,
                drone_position,
                self.camera_params
            )
            
            all_detected_targets.extend(detected_targets)
            
            # Проверяем цели, уже назначенные дрону
            if drone.state in [DroneState.TARGET_TRACKING, DroneState.ATTACK]:
                if drone.target_position is not None:
                    # Если дрон уже атакует цель, проверяем завершение атаки
                    if drone.state == DroneState.ATTACK:
                        if drone.attack_target(drone.target_position):
                            # Атака успешна
                            self._log(f"Drone {drone_id} successfully attacked target")
                            self.stats['targets_attacked'] += 1
                            
                            # Сбрасываем цель дрона
                            drone.target_position = None
                            drone.state = DroneState.PATROL
        
        # Треккинг целей
        tracked_targets = self.target_detector.track_targets(all_detected_targets)
        self.stats['targets_detected'] = len(self.target_detector.tracked_targets)
        
        # Обновляем цели в контроллере роя для распределения
        for target_id, target in tracked_targets.items():
            self.swarm_controller.add_target(
                target_id,
                target['position'],
                target['class']
            )

    def _create_drone_camera_frame(self, drone_id: int) -> np.ndarray:
        """Создание виртуального кадра с камеры дрона"""
        # Создаем чистый кадр
        frame = np.zeros((
            self.camera_params['resolution'][1],
            self.camera_params['resolution'][0],
            3
        ), dtype=np.uint8)
        
        # Добавляем фоновую текстуру (земля, небо)
        self._add_background_texture(frame, self.drones[drone_id].position[2])
        
        # В будущем здесь можно добавить рендеринг целей в поле зрения дрона
        
        return frame

    def _add_background_texture(self, frame: np.ndarray, altitude: float):
        """Добавление фоновой текстуры для симуляции изображения"""
        # Определяем горизонт (зависит от высоты дрона)
        horizon_y = int(frame.shape[0] * (0.5 - altitude / 3000))
        horizon_y = max(0, min(horizon_y, frame.shape[0]))
        
        # Небо (синий)
        frame[:horizon_y, :] = np.array([200, 150, 100])
        
        # Земля (зеленый/коричневый)
        frame[horizon_y:, :] = np.array([100, 150, 100])
        
        # Добавляем шум для текстуры
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Добавляем сетку для земли (имитация ландшафта)
        if horizon_y < frame.shape[0]:
            grid_size = 50
            for i in range(horizon_y, frame.shape[0], grid_size):
                cv2.line(frame, (0, i), (frame.shape[1], i), (80, 120, 80), 1)
            
            for i in range(0, frame.shape[1], grid_size):
                cv2.line(frame, (i, horizon_y), (i, frame.shape[0]), (80, 120, 80), 1)

    def _drone_simulation_loop(self):
        """Цикл симуляции физики дронов"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            for drone_id, drone in self.drones.items():
                # Обновляем состояние дрона из контроллера
                controller_drone = self.swarm_controller.drones[drone_id]
                drone.state = DroneState(controller_drone['state'])
                
                # Обновление в зависимости от состояния
                if drone.state == DroneState.PATROL:
                    drone.update_patrol()
                elif drone.state == DroneState.TARGET_TRACKING:
                    # Если дрону назначена цель в контроллере, но он ее еще не отслеживает
                    if controller_drone['target_id'] is not None and drone.target_position is None:
                        target = self.swarm_controller.targets[controller_drone['target_id']]
                        drone.set_target(target['position'])
                    
                    # Обновляем движение к цели
                    if drone.target_position is not None:
                        drone.move_to(drone.target_position)
                elif drone.state == DroneState.RTB:
                    drone.return_to_base()
                
                # Физическое обновление дрона
                drone.update(dt)
                
                # Синхронизация положения с контроллером
                self.swarm_controller.update_drone_position(
                    drone_id, 
                    drone.position,
                    drone.velocity
                )
            
            time.sleep(0.01)  # 100Hz physics update

    def _visualization_loop(self):
        """Цикл визуализации состояния роя"""
        while self.running:
            # Создание изображения для карты
            map_image = self._create_map_visualization()
            
            # Создание изображения камеры выбранного дрона
            camera_view = self._create_selected_drone_camera_view()
            
            # Отображение визуализаций
            cv2.imshow('Drone Swarm Control', map_image)
            cv2.imshow('Drone Camera View', camera_view)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
                break
            elif key == ord('n'):
                # Переключение на следующий дрон для просмотра
                self.selected_drone_id = (self.selected_drone_id + 1) % len(self.drones)
            
            time.sleep(0.03)  # ~30fps

    def _create_map_visualization(self) -> np.ndarray:
        """Создание карты для визуализации роя дронов"""
        # Создание базового изображения (черный фон)
        map_image = np.zeros((self.map_size[1], self.map_size[0], 3), dtype=np.uint8)
        
        # Рисуем координатную сетку
        self._draw_grid(map_image)
        
        # Рисуем все синтетические цели
        self._draw_all_targets(map_image)
        
        # Рисуем дронов
        self._draw_drones(map_image)
        
        # Отображение статуса
        self._draw_status(map_image)
        
        return map_image

    def _draw_grid(self, image: np.ndarray):
        """Рисуем координатную сетку на карте"""
        # Рисуем основные оси
        cv2.line(image, 
                (0, int(self.center_offset[1])), 
                (self.map_size[0], int(self.center_offset[1])), 
                (50, 50, 50), 1)
        cv2.line(image, 
                (int(self.center_offset[0]), 0), 
                (int(self.center_offset[0]), self.map_size[1]), 
                (50, 50, 50), 1)
        
        # Рисуем сетку (каждые 500 метров)
        grid_size = 500
        grid_pixels = int(grid_size * self.scale_factor)
        
        # Вертикальные линии
        for x in range(int(self.center_offset[0]), 0, -grid_pixels):
            cv2.line(image, (x, 0), (x, self.map_size[1]), (20, 20, 20), 1)
            label = f"{int((x - self.center_offset[0]) / self.scale_factor)}m"
            cv2.putText(image, label, (x, self.center_offset[1] + 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
        for x in range(int(self.center_offset[0]) + grid_pixels, self.map_size[0], grid_pixels):
            cv2.line(image, (x, 0), (x, self.map_size[1]), (20, 20, 20), 1)
            label = f"{int((x - self.center_offset[0]) / self.scale_factor)}m"
            cv2.putText(image, label, (x, self.center_offset[1] + 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
        # Горизонтальные линии
        for y in range(int(self.center_offset[1]), 0, -grid_pixels):
            cv2.line(image, (0, y), (self.map_size[0], y), (20, 20, 20), 1)
            label = f"{int((self.center_offset[1] - y) / self.scale_factor)}m"
            cv2.putText(image, label, (self.center_offset[0] + 5, y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
        for y in range(int(self.center_offset[1]) + grid_pixels, self.map_size[1], grid_pixels):
            cv2.line(image, (0, y), (self.map_size[0], y), (20, 20, 20), 1)
            label = f"{int((self.center_offset[1] - y) / self.scale_factor)}m"
            cv2.putText(image, label, (self.center_offset[0] + 5, y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _draw_all_targets(self, image: np.ndarray):
        """Рисуем все цели на карте"""
        # Получаем все синтетические цели
        all_targets = self.target_detector.get_synthetic_targets_for_visualization()
        
        for target in all_targets:
            # Конвертируем координаты в пиксели
            px, py = self._world_to_pixel(target['position'])
            
            # Цвет в зависимости от типа цели
            color = {
                'person': (0, 255, 0),    # Зеленый
                'car': (255, 255, 0),     # Желтый
                'truck': (255, 128, 0),   # Оранжевый
                'tank': (0, 0, 255)       # Красный
            }.get(target['class'], (0, 255, 0))
            
            # Размер в зависимости от типа цели
            size = {
                'person': 3,
                'car': 5,
                'truck': 7,
                'tank': 8
            }.get(target['class'], 5)
            
            # Отрисовка цели
            cv2.circle(image, (px, py), size, color, -1)
            
            # Отображение ID и класса цели
            label = f"{target['id']}: {target['class']}"
            cv2.putText(image, label, (px + 5, py - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Если цель назначена дрону, рисуем линию к соответствующему дрону
            if target['id'] in self.swarm_controller.targets:
                ctrl_target = self.swarm_controller.targets[target['id']]
                if ctrl_target.get('assigned_drone') is not None:
                    drone_id = ctrl_target['assigned_drone']
                    drone_pos = self.drones[drone_id].position
                    drone_px, drone_py = self._world_to_pixel(drone_pos)
                    
                    # Рисуем линию между дроном и целью
                    cv2.line(image, (px, py), (drone_px, drone_py), color, 1)

    def _draw_drones(self, image: np.ndarray):
        """Рисуем все дроны на карте"""
        for drone_id, drone in self.drones.items():
            # Конвертируем координаты в пиксели
            px, py = self._world_to_pixel(drone.position)
            
            # Цвет в зависимости от состояния
            color = {
                DroneState.IDLE: (128, 128, 128),            # Серый
                DroneState.TAKEOFF: (128, 128, 255),         # Светло-синий
                DroneState.PATROL: (0, 255, 0),              # Зеленый
                DroneState.TARGET_TRACKING: (255, 255, 0),   # Желтый
                DroneState.ATTACK: (0, 0, 255),              # Красный
                DroneState.RTB: (255, 0, 255)                # Пурпурный
            }.get(drone.state, (255, 255, 255))
            
            # Выделение выбранного дрона
            if drone_id == self.selected_drone_id:
                cv2.circle(image, (px, py), 12, (255, 255, 255), 1)
            
            # Отрисовка дрона
            cv2.circle(image, (px, py), 8, color, -1)
            
            # Направление движения (вектор скорости)
            if np.linalg.norm(drone.velocity) > 0.1:
                velocity_norm = drone.velocity / np.linalg.norm(drone.velocity) * 20
                end_point = (
                    int(px + velocity_norm[0]),
                    int(py - velocity_norm[1])  # Инвертируем Y для отображения
                )
                cv2.arrowedLine(image, (px, py), end_point, color, 2)
            
            # ID дрона
            cv2.putText(image, str(drone_id), (px - 3, py + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Информация о дроне (пройденный путь, батарея)
            info_text = f"D:{drone.distance_traveled/1000:.1f}km B:{drone.battery_level:.0f}%"
            cv2.putText(image, info_text, (px + 10, py), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Если у дрона есть цель, но линия еще не нарисована, рисуем ее
            if drone.target_position is not None:
                target_px, target_py = self._world_to_pixel(drone.target_position)
                cv2.line(image, (px, py), (target_px, target_py), color, 1, cv2.LINE_AA)

    def _draw_status(self, image: np.ndarray):
        """Отображение статуса системы на карте"""
        # Создаем панель статуса
        status_height = 130
        status_panel = np.zeros((status_height, image.shape[1], 3), dtype=np.uint8)
        
        # Соединяем с основным изображением
        combined_image = np.vstack((image, status_panel))
        
        # Информация о миссии
        mission_time = time.time() - self.stats['mission_start_time']
        mission_info = [
            f"Mission Time: {int(mission_time//3600):02d}:{int((mission_time%3600)//60):02d}:{int(mission_time%60):02d}",
            f"Drones: {sum(1 for d in self.drones.values() if d.battery_level > 0)}/{len(self.drones)}",
            f"Mode: {'ATTACK' if self.attack_mode else 'PATROL'}",
            f"Targets Detected: {self.stats['targets_detected']}",
            f"Targets Attacked: {self.stats['targets_attacked']}",
            f"Max Distance: {self.stats['max_distance_traveled']/1000:.2f} km",
            f"Attack Threshold: {self.swarm_controller.attack_distance_threshold/1000:.1f} km"
        ]
        
        for i, text in enumerate(mission_info):
            y_pos = image.shape[0] + 20 + i * 15
            cv2.putText(combined_image, text, (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Командные подсказки
        commands = [
            "Commands: [attack] - Attack Mode, [patrol] - Patrol Mode,",
            "[status] - Show Status, [stop] - Emergency Stop, [quit] - Exit"
        ]
        
        for i, text in enumerate(commands):
            y_pos = image.shape[0] + 20 + i * 15
            cv2.putText(combined_image, text, (400, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return combined_image

    def _create_selected_drone_camera_view(self) -> np.ndarray:
        """Создание изображения с камеры выбранного дрона"""
        # Проверка наличия выбранного дрона
        if self.selected_drone_id not in self.drones:
            return np.zeros((
                self.camera_params['resolution'][1],
                self.camera_params['resolution'][0],
                3
            ), dtype=np.uint8)
        
        # Создаем кадр с камеры
        frame = self._create_drone_camera_frame(self.selected_drone_id)
        
        # Получаем текущее время
        current_time = time.time()
        
        # Отображаем информацию о дроне
        drone = self.drones[self.selected_drone_id]
        info_text = [
            f"Drone ID: {self.selected_drone_id}",
            f"State: {drone.state.value}",
            f"Position: ({drone.position[0]:.1f}, {drone.position[1]:.1f}, {drone.position[2]:.1f})",
            f"Speed: {np.linalg.norm(drone.velocity):.1f} m/s",
            f"Battery: {drone.battery_level:.1f}%",
            f"Distance: {drone.distance_traveled/1000:.2f} km"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Добавляем HUD
        self._draw_drone_hud(frame, drone)
        
        # Обнаружение целей в поле зрения дрона
        detected_targets = self.target_detector.detect_targets(
            frame,
            drone.position,
            self.camera_params
        )
        
        # Отрисовка целей на кадре
        if detected_targets:
            frame = self.target_detector.draw_targets(frame, 
                {t['id']: t for t in detected_targets}
            )
        
        return frame

    def _draw_drone_hud(self, frame: np.ndarray, drone: Drone):
        """Отрисовка HUD (Heads-Up Display) для дрона"""
        h, w = frame.shape[:2]
        
        # Горизонтальная линия по центру
        cv2.line(frame, (0, h//2), (w, h//2), (255, 255, 255), 1)
        
        # Вертикальная линия по центру
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 1)
        
        # Перекрестие в центре
        cv2.circle(frame, (w//2, h//2), 5, (0, 255, 0), 1)
        cv2.circle(frame, (w//2, h//2), 2, (0, 255, 0), -1)
        
        # Компас (отображение направления)
        heading = int(np.degrees(np.arctan2(drone.velocity[0], drone.velocity[1])) % 360)
        cv2.putText(frame, f"HDG: {heading:03d}", (w-150, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Высота
        cv2.putText(frame, f"ALT: {drone.position[2]:.1f}m", (w-150, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Скорость
        speed = np.linalg.norm(drone.velocity)
        cv2.putText(frame, f"SPD: {speed:.1f}m/s", (w-150, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Режим дрона
        mode_color = {
            DroneState.PATROL: (0, 255, 0),         # Зеленый
            DroneState.TARGET_TRACKING: (0, 255, 255), # Желтый
            DroneState.ATTACK: (0, 0, 255),         # Красный
            DroneState.RTB: (255, 0, 255)           # Пурпурный
        }.get(drone.state, (255, 255, 255))
        
        cv2.putText(frame, f"MODE: {drone.state.value}", (10, h-30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # Если дрон атакует цель, показываем индикатор цели
        if drone.state in [DroneState.TARGET_TRACKING, DroneState.ATTACK] and drone.target_position is not None:
            # Проекция цели на экран
            target_pixel = self._project_3d_to_2d(
                drone.target_position,
                drone.position,
                self.camera_params
            )
            
            # Если цель в поле зрения, рисуем наводку
            if (0 <= target_pixel[0] < w and 0 <= target_pixel[1] < h):
                cv2.circle(frame, (int(target_pixel[0]), int(target_pixel[1])), 
                          20, (0, 0, 255), 2)
                
                # Расстояние до цели
                distance = np.linalg.norm(drone.position - drone.target_position)
                cv2.putText(frame, f"TGT: {distance:.1f}m", (int(target_pixel[0]) + 25, int(target_pixel[1])), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _project_3d_to_2d(self, target_position: np.ndarray, drone_position: np.ndarray, camera_params: Dict) -> Tuple[float, float]:
        """Проекция 3D координат цели на 2D плоскость экрана"""
        # Используем готовую функцию из target_detector
        return self.target_detector._project_3d_to_2d(
            target_position,
            drone_position,
            camera_params
        )

    def _world_to_pixel(self, position: np.ndarray) -> Tuple[int, int]:
        """Конвертация мировых координат в координаты на экране"""
        px = int(position[0] * self.scale_factor + self.center_offset[0])
        py = int(self.center_offset[1] - position[1] * self.scale_factor)  # Инвертируем Y
        return (px, py)

    def _operator_interface(self):
        """Интерфейс оператора"""
        print("\nDrone Swarm Control Interface")
        print("Commands:")
        print("1. 'attack' - Enable attack mode")
        print("2. 'patrol' - Enable patrol mode")
        print("3. 'status' - Show swarm status")
        print("4. 'stop' - Emergency stop")
        print("5. 'quit' - Exit program")
        print("6. 'select <id>' - View from drone camera")
        
        while self.running:
            cmd = input("\nEnter command: ").lower().strip()
            
            if cmd == 'attack':
                self.attack_mode = True
                self.swarm_controller.set_attack_mode(True)
                self._log("Attack mode enabled")
                print("Attack mode enabled")
                
            elif cmd == 'patrol':
                self.attack_mode = False
                self.swarm_controller.set_attack_mode(False)
                self._log("Patrol mode enabled")
                print("Patrol mode enabled")
                
            elif cmd == 'status':
                status = self.swarm_controller.get_swarm_status()
                print("\nSwarm Status:")
                for key, value in status.items():
                    print(f"{key}: {value}")
                    
            elif cmd == 'stop':
                self.swarm_controller.emergency_stop()
                self._log("Emergency stop initiated")
                print("Emergency stop initiated")
                
            elif cmd.startswith('select '):
                try:
                    drone_id = int(cmd.split(' ')[1])
                    if drone_id in self.drones:
                        self.selected_drone_id = drone_id
                        print(f"Selected drone {drone_id} for camera view")
                    else:
                        print(f"Error: Drone {drone_id} not found")
                except (ValueError, IndexError):
                    print("Error: Invalid drone ID")
                
            elif cmd == 'quit':
                self.stop()
                break
                
            else:
                print("Unknown command")

    def _log(self, message: str):
        """Запись сообщения в лог"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)

    def stop(self):
        """Остановка системы"""
        self.running = False
        self.swarm_controller.emergency_stop()
        
        self._log("System stopped")
        print("System stopped")
        
        # Закрываем окна визуализации
        cv2.destroyAllWindows()
        
        # Сохраняем итоговую статистику
        self._log(f"Final statistics: {self.stats}")

def main():
    parser = argparse.ArgumentParser(description='Drone Swarm Control System')
    parser.add_argument('--num-drones', type=int, default=10,
                      help='Number of drones in the swarm')
    args = parser.parse_args()
    
    system = DroneSwarmSystem(args.num_drones)
    try:
        system.start()
    except KeyboardInterrupt:
        system.stop()
        print("\nSystem shutdown by operator")

if __name__ == "__main__":
    main() 