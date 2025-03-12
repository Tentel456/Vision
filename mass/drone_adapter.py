import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
import threading
import os
import cv2

# Импорт системы коннекторов для дронов
from drone_connectors import get_connector, DroneConnector

class DroneAdapter:
    """
    Адаптер для подключения реальных дронов к системе управления роем.
    Обеспечивает унифицированный интерфейс для взаимодействия с физическими дронами.
    """
    
    def __init__(self, drone_id: int, drone_type: str = None, connection_params: Dict = None):
        """
        Инициализация адаптера для дрона
        
        Args:
            drone_id: Идентификатор дрона
            drone_type: Тип дрона ('dji', 'mavlink', 'ardupilot', 'px4', 'parrot')
            connection_params: Параметры подключения к дрону
        """
        self.id = drone_id
        self.drone_type = drone_type
        self.connection_params = connection_params or {}
        self.connector = None
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)  # roll, pitch, yaw
        self.battery_level = 100.0
        self.distance_traveled = 0.0
        self.previous_position = np.zeros(3)
        self.is_simulated = True
        self.target_position = None
        self.state = "IDLE"
        self.is_active = True
        self.patrol_waypoints = []
        self.current_waypoint_index = 0
        
        # Настройка логгирования
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'drone_{drone_id}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"DroneAdapter-{drone_id}")
        
        # Если указан тип дрона, создаем коннектор
        if drone_type:
            self.is_simulated = False
            try:
                self.connector = get_connector(drone_type, **self.connection_params)
                self.logger.info(f"Создан коннектор для дрона типа {drone_type}")
            except Exception as e:
                self.logger.error(f"Ошибка создания коннектора: {str(e)}")
                self.is_simulated = True
        
        # Если используется симуляция, инициализируем параметры
        if self.is_simulated:
            self.logger.info("Используется симуляция дрона")
            
            # Случайное начальное положение для симуляции
            self.position = np.array([
                np.random.uniform(-50, 50),  # X
                np.random.uniform(-50, 50),  # Y
                np.random.uniform(30, 50)    # Z (высота)
            ])
            self.previous_position = self.position.copy()
            
            # Параметры симулированного дрона
            self.max_speed = 15.0  # м/с
            self.max_acceleration = 5.0  # м/с^2
            
            # Границы полета
            self.flight_boundaries = {
                'x_min': -3000.0,
                'x_max': 3000.0,
                'y_min': -3000.0,
                'y_max': 3000.0,
                'z_min': 10.0,    # Минимальная высота
                'z_max': 500.0    # Максимальная высота
            }
            
            # Параметры для предотвращения столкновений
            self.collision_avoidance_radius = 30.0  # метры
            self.nearby_drones = []  # Список ближайших дронов
    
    def connect(self) -> bool:
        """
        Подключение к дрону или инициализация симуляции
        
        Returns:
            bool: True если подключение успешно, иначе False
        """
        if self.is_simulated:
            self.logger.info("Симуляция дрона инициализирована")
            return True
            
        try:
            if self.connector:
                success = self.connector.connect(self.connection_params)
                if success:
                    self.logger.info("Подключение к дрону успешно установлено")
                    
                    # Получение начальных данных от дрона
                    self.position = self.connector.get_position()
                    self.velocity = self.connector.get_velocity()
                    self.orientation = np.array(self.connector.get_attitude())
                    self.battery_level = self.connector.get_battery_level()
                    self.previous_position = self.position.copy()
                    
                    return True
                else:
                    self.logger.error("Не удалось подключиться к дрону")
                    self.is_simulated = True
                    return False
            else:
                self.logger.error("Коннектор не инициализирован")
                self.is_simulated = True
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка подключения к дрону: {str(e)}")
            self.is_simulated = True
            return False
    
    def disconnect(self) -> bool:
        """
        Отключение от дрона
        
        Returns:
            bool: True если отключение успешно, иначе False
        """
        if self.is_simulated:
            self.logger.info("Симуляция дрона завершена")
            return True
            
        try:
            if self.connector:
                success = self.connector.disconnect()
                if success:
                    self.logger.info("Отключение от дрона выполнено успешно")
                    return True
                else:
                    self.logger.error("Ошибка при отключении от дрона")
                    return False
            else:
                self.logger.warning("Коннектор не инициализирован")
                return True
                
        except Exception as e:
            self.logger.error(f"Ошибка отключения от дрона: {str(e)}")
            return False
    
    def takeoff(self, altitude: float = 50.0) -> bool:
        """
        Взлет дрона на указанную высоту
        
        Args:
            altitude: Высота взлета в метрах
            
        Returns:
            bool: True если взлет успешен, иначе False
        """
        self.state = "TAKEOFF"
        
        if self.is_simulated:
            self.logger.info(f"Симуляция взлета дрона на высоту {altitude} м")
            self.position[2] = altitude
            time.sleep(1)  # Имитация времени взлета
            self.state = "PATROL"
            return True
            
        try:
            if self.connector:
                success = self.connector.takeoff(altitude)
                if success:
                    self.logger.info(f"Взлет на высоту {altitude} м выполнен успешно")
                    self.state = "PATROL"
                    return True
                else:
                    self.logger.error("Ошибка при взлете")
                    self.state = "IDLE"
                    return False
            else:
                self.logger.error("Коннектор не инициализирован")
                self.state = "IDLE"
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка взлета: {str(e)}")
            self.state = "IDLE"
            return False
    
    def land(self) -> bool:
        """
        Посадка дрона
        
        Returns:
            bool: True если посадка успешна, иначе False
        """
        self.state = "RTB"
        
        if self.is_simulated:
            self.logger.info("Симуляция посадки дрона")
            self.position[2] = 0
            time.sleep(1)  # Имитация времени посадки
            self.state = "IDLE"
            return True
            
        try:
            if self.connector:
                success = self.connector.land()
                if success:
                    self.logger.info("Посадка выполнена успешно")
                    self.state = "IDLE"
                    return True
                else:
                    self.logger.error("Ошибка при посадке")
                    return False
            else:
                self.logger.error("Коннектор не инициализирован")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка посадки: {str(e)}")
            return False
    
    def return_to_base(self) -> bool:
        """
        Возврат дрона на базу
        
        Returns:
            bool: True если команда успешно отправлена, иначе False
        """
        self.state = "RTB"
        
        if self.is_simulated:
            self.logger.info("Симуляция возврата дрона на базу")
            self.position = np.zeros(3)
            self.position[2] = 30  # Высота возврата
            time.sleep(1)  # Имитация времени возврата
            return True
            
        try:
            if self.connector:
                success = self.connector.return_to_home()
                if success:
                    self.logger.info("Команда возврата на базу отправлена успешно")
                    return True
                else:
                    self.logger.error("Ошибка при отправке команды возврата на базу")
                    return False
            else:
                self.logger.error("Коннектор не инициализирован")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка возврата на базу: {str(e)}")
            return False
    
    def move_to(self, target_position: np.ndarray, max_speed: Optional[float] = None) -> bool:
        """
        Перемещение дрона в указанную позицию
        
        Args:
            target_position: Целевая позиция [x, y, z] в метрах
            max_speed: Максимальная скорость перемещения в м/с
            
        Returns:
            bool: True если команда успешно отправлена, иначе False
        """
        if self.is_simulated:
            # В симуляции просто сохраняем целевую позицию
            self.target_position = target_position
            direction = target_position - self.position
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:  # Если расстояние существенное
                # Нормализованный вектор направления
                direction = direction / distance
                
                # Расчет желаемой скорости
                speed = max_speed if max_speed is not None else self.max_speed
                desired_velocity = direction * speed
                
                # Обновление скорости для симуляции
                self.velocity = desired_velocity
                
            self.logger.info(f"Симуляция перемещения дрона к {target_position}")
            return True
            
        try:
            if self.connector:
                success = self.connector.move_to(target_position, max_speed)
                if success:
                    self.logger.info(f"Команда перемещения к {target_position} отправлена успешно")
                    self.target_position = target_position
                    return True
                else:
                    self.logger.error("Ошибка при отправке команды перемещения")
                    return False
            else:
                self.logger.error("Коннектор не инициализирован")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка перемещения: {str(e)}")
            return False
    
    def set_velocity(self, velocity: np.ndarray) -> bool:
        """
        Установка скорости дрона
        
        Args:
            velocity: Вектор скорости [vx, vy, vz] в м/с
            
        Returns:
            bool: True если команда успешно отправлена, иначе False
        """
        if self.is_simulated:
            self.velocity = velocity
            return True
            
        try:
            if self.connector:
                success = self.connector.set_velocity(velocity)
                if success:
                    self.logger.info(f"Команда установки скорости {velocity} отправлена успешно")
                    return True
                else:
                    self.logger.error("Ошибка при отправке команды установки скорости")
                    return False
            else:
                self.logger.error("Коннектор не инициализирован")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка установки скорости: {str(e)}")
            return False
    
    def update(self, dt: float):
        """
        Обновление состояния дрона
        
        Args:
            dt: Время с предыдущего обновления в секундах
        """
        # Сохраняем текущую позицию для расчета пройденного расстояния
        old_position = self.position.copy()
        
        if self.is_simulated:
            # Обновление позиции в симуляции
            self.position += self.velocity * dt
            
            # Случайная турбулентность для реалистичности
            self.position += np.array([
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.05, 0.05)
            ])
            
            # Проверка и корректировка границ полета
            self._check_boundaries()
            
            # Применяем алгоритм предотвращения столкновений в симуляции
            if hasattr(self, 'nearby_drones') and self.nearby_drones:
                self._apply_collision_avoidance()
                
            # Симуляция разрядки батареи
            speed = np.linalg.norm(self.velocity)
            battery_drain = dt * (0.1 + 0.1 * speed / self.max_speed)
            
            # Дополнительный расход при атаке
            if self.state == "ATTACK":
                battery_drain *= 1.5
                
            self.battery_level = max(0.0, self.battery_level - battery_drain)
            
            # Автоматическое возвращение на базу при низком заряде
            if self.battery_level < 20.0 and self.state != "RTB":
                self.return_to_base()
                self.logger.info(f"Низкий заряд батареи ({self.battery_level:.1f}%). Возврат на базу.")
                
        else:
            # Обновление данных от реального дрона
            if self.connector:
                try:
                    self.position = self.connector.get_position()
                    self.velocity = self.connector.get_velocity()
                    self.orientation = np.array(self.connector.get_attitude())
                    self.battery_level = self.connector.get_battery_level()
                except Exception as e:
                    self.logger.error(f"Ошибка получения данных от дрона: {str(e)}")
        
        # Обновляем пройденное расстояние
        traveled = np.linalg.norm(self.position - old_position)
        self.distance_traveled += traveled
        self.previous_position = old_position
    
    def _check_boundaries(self):
        """
        Проверка и корректировка границ полета (только для симуляции)
        """
        if not self.is_simulated:
            return
            
        # Проверка X границ
        if self.position[0] < self.flight_boundaries['x_min']:
            self.position[0] = self.flight_boundaries['x_min']
            self.velocity[0] = max(0, self.velocity[0])  # Не позволяем двигаться дальше за границу
        elif self.position[0] > self.flight_boundaries['x_max']:
            self.position[0] = self.flight_boundaries['x_max']
            self.velocity[0] = min(0, self.velocity[0])
        
        # Проверка Y границ
        if self.position[1] < self.flight_boundaries['y_min']:
            self.position[1] = self.flight_boundaries['y_min']
            self.velocity[1] = max(0, self.velocity[1])
        elif self.position[1] > self.flight_boundaries['y_max']:
            self.position[1] = self.flight_boundaries['y_max']
            self.velocity[1] = min(0, self.velocity[1])
        
        # Проверка Z границ (высота)
        if self.position[2] < self.flight_boundaries['z_min']:
            self.position[2] = self.flight_boundaries['z_min']
            self.velocity[2] = max(0, self.velocity[2])
        elif self.position[2] > self.flight_boundaries['z_max']:
            self.position[2] = self.flight_boundaries['z_max']
            self.velocity[2] = min(0, self.velocity[2])
    
    def set_nearby_drones(self, drone_positions: List[Tuple[int, np.ndarray]]):
        """
        Обновление информации о ближайших дронах для предотвращения столкновений
        
        Args:
            drone_positions: Список кортежей (id, position) для ближайших дронов
        """
        if not self.is_simulated:
            return
            
        self.nearby_drones = []
        for drone_id, position in drone_positions:
            if drone_id != self.id:  # Исключаем себя
                distance = np.linalg.norm(position - self.position)
                if distance < self.collision_avoidance_radius:
                    self.nearby_drones.append((drone_id, position, distance))
    
    def _apply_collision_avoidance(self):
        """
        Применение алгоритма предотвращения столкновений (только для симуляции)
        """
        if not self.is_simulated:
            return
            
        avoidance_force = np.zeros(3)
        
        for drone_id, position, distance in self.nearby_drones:
            # Направление от ближайшего дрона к текущему
            direction = self.position - position
            
            # Нормализация направления
            if distance > 0:
                direction = direction / distance
            else:
                # Если дроны в одной точке, добавляем случайное смещение
                direction = np.array([
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1)
                ])
                
            # Сила отталкивания обратно пропорциональна расстоянию
            strength = 1.0 - distance / self.collision_avoidance_radius
            avoidance_force += direction * strength * 5.0  # Коэффициент силы
        
        # Применяем силу отталкивания к скорости
        self.velocity += avoidance_force
        
        # Ограничиваем скорость
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
    
    def start_patrol(self, waypoints: List[np.ndarray] = None):
        """
        Начало патрулирования по заданным точкам
        
        Args:
            waypoints: Список точек патрулирования [np.ndarray] в формате [x, y, z]
        """
        if waypoints is None or len(waypoints) == 0:
            # Если точки не предоставлены, создаем случайные вокруг текущей позиции
            waypoints = self._generate_random_patrol_points()
            
        self.patrol_waypoints = waypoints
        self.current_waypoint_index = 0
        self.state = "PATROL"
        
        if not self.is_simulated and self.connector:
            # В реальном дроне просто начинаем движение к первой точке
            self.move_to(waypoints[0])
    
    def _generate_random_patrol_points(self, num_points: int = 5, radius: float = 500.0) -> List[np.ndarray]:
        """
        Генерация случайных точек патрулирования вокруг текущей позиции
        
        Args:
            num_points: Количество точек
            radius: Радиус зоны патрулирования
            
        Returns:
            List[np.ndarray]: Список точек патрулирования
        """
        waypoints = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            z = self.position[2] + np.random.uniform(-50, 50)
            
            # Обеспечиваем, что высота в допустимых пределах
            if self.is_simulated:
                z = max(self.flight_boundaries['z_min'], min(z, self.flight_boundaries['z_max']))
            else:
                # Примерные ограничения для реальных дронов
                z = max(20.0, min(z, 120.0))
            
            waypoint = self.position + np.array([dx, dy, 0])
            waypoint[2] = z
            waypoints.append(waypoint)
            
        return waypoints
    
    def update_patrol(self):
        """
        Обновление патрулирования
        """
        if not self.patrol_waypoints:
            # Если нет точек патрулирования, создаем их
            self.patrol_waypoints = self._generate_random_patrol_points()

        current_waypoint = self.patrol_waypoints[self.current_waypoint_index]
        distance = np.linalg.norm(current_waypoint - self.position)

        if distance < 2.0:  # Если достигли текущей точки
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.patrol_waypoints)
            current_waypoint = self.patrol_waypoints[self.current_waypoint_index]
            self.logger.info(f"Достигнута точка маршрута {self.current_waypoint_index-1}. "
                          f"Пройденное расстояние: {self.distance_traveled/1000:.2f} км")
            
            # Перемещение к следующей точке
            self.move_to(current_waypoint, max_speed=self.max_speed * 0.7 if self.is_simulated else None)
        elif self.is_simulated:
            # В симуляции обновляем движение к текущей точке
            self.move_to(current_waypoint, max_speed=self.max_speed * 0.7)
    
    def set_target(self, target_position: np.ndarray):
        """
        Установка цели для отслеживания или атаки
        
        Args:
            target_position: Позиция цели [x, y, z] в метрах
        """
        self.target_position = target_position
        self.state = "TARGET_TRACKING"
        
        # Перемещение к цели
        self.move_to(target_position)
    
    def attack_target(self, target_position: np.ndarray) -> bool:
        """
        Атака цели
        
        Args:
            target_position: Позиция цели [x, y, z] в метрах
            
        Returns:
            bool: True если атака успешна, иначе False
        """
        if self.state != "ATTACK":
            self.state = "ATTACK"
            self.target_position = target_position

        # Расчет расстояния до цели
        distance = np.linalg.norm(target_position - self.position)
        attack_range = 20.0  # Дальность атаки
        
        if distance <= attack_range:
            # Выполнение атаки
            # Имитация времени, необходимого для атаки
            time_to_attack = 3.0  # секунды
            
            # Возвращаем True, если атака успешна (в реальной системе будет более сложная логика)
            attack_success = np.random.random() < 0.7  # 70% шанс успешной атаки
            
            if attack_success:
                self.logger.info(f"Успешная атака цели в позиции {target_position}!")
                
            return attack_success
        else:
            # Движение к цели
            self.move_to(target_position)
            return False
    
    def get_video_frame(self) -> Optional[np.ndarray]:
        """
        Получение кадра с камеры дрона
        
        Returns:
            Optional[np.ndarray]: Изображение с камеры дрона или None
        """
        if self.is_simulated:
            # Создаем симулированный кадр
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Добавляем фоновую текстуру (земля, небо)
            horizon_y = int(frame.shape[0] * (0.5 - self.position[2] / 3000))
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
            
            return frame
        else:
            # Получение кадра с реального дрона
            if self.connector:
                try:
                    return self.connector.get_video_frame()
                except Exception as e:
                    self.logger.error(f"Ошибка получения видео: {str(e)}")
                    return None
            else:
                return None
    
    def emergency_stop(self) -> bool:
        """
        Экстренная остановка дрона
        
        Returns:
            bool: True если команда успешно отправлена, иначе False
        """
        self.state = "RTB"
        
        if self.is_simulated:
            self.velocity = np.zeros(3)
            self.logger.warning("ЭКСТРЕННАЯ ОСТАНОВКА ДРОНА (СИМУЛЯЦИЯ)")
            return True
            
        try:
            if self.connector:
                success = self.connector.emergency_stop()
                if success:
                    self.logger.warning("ЭКСТРЕННАЯ ОСТАНОВКА ДРОНА ВЫПОЛНЕНА")
                    return True
                else:
                    self.logger.error("Ошибка при выполнении экстренной остановки")
                    return False
            else:
                self.logger.error("Коннектор не инициализирован")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка экстренной остановки: {str(e)}")
            return False
    
    def get_status(self) -> Dict:
        """
        Получение текущего статуса дрона
        
        Returns:
            Dict: Словарь с параметрами состояния дрона
        """
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'state': self.state,
            'battery': self.battery_level,
            'distance_traveled': self.distance_traveled,
            'target_position': self.target_position.tolist() if self.target_position is not None else None,
            'is_simulated': self.is_simulated
        } 