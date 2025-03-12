import numpy as np
from typing import Tuple, Optional, List
import time
from enum import Enum
import random

class DroneState(Enum):
    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    PATROL = "PATROL"
    TARGET_TRACKING = "TARGET_TRACKING"
    ATTACK = "ATTACK"
    RTB = "RTB"  # Return to base

class Drone:
    def __init__(self, drone_id: int, initial_position: np.ndarray = None):
        self.id = drone_id
        self.position = initial_position if initial_position is not None else np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.orientation = np.zeros(3)  # roll, pitch, yaw
        self.state = DroneState.IDLE
        self.target_position = None
        self.battery_level = 100
        self.max_speed = 15.0  # m/s
        self.max_acceleration = 5.0  # m/s^2
        self.detection_range = 100.0  # meters
        self.attack_range = 50.0  # meters
        self.last_update = time.time()
        self.home_position = np.copy(self.position)
        self.patrol_waypoints = []
        self.current_waypoint_index = 0
        self.distance_traveled = 0.0
        self.previous_position = np.copy(self.position)
        
        # Безопасные границы полета
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

    def update(self, dt: float):
        """Обновление состояния дрона"""
        # Сохраняем текущую позицию для расчета пройденного расстояния
        old_position = np.copy(self.position)
        
        # Обновляем позицию
        self._update_position(dt)
        
        # Проверяем и корректируем границы полета
        self._check_boundaries()
        
        # Применяем алгоритм предотвращения столкновений
        if self.nearby_drones:
            self._apply_collision_avoidance()
        
        # Обновляем пройденное расстояние
        traveled = np.linalg.norm(self.position - old_position)
        self.distance_traveled += traveled
        
        # Обновляем батарею
        self._update_battery(dt)

    def _update_position(self, dt: float):
        """Обновление позиции дрона на основе физики"""
        # Обновление скорости
        self.velocity += self.acceleration * dt
        
        # Ограничение скорости
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        
        # Обновление позиции
        self.position += self.velocity * dt
        
        # Случайная турбулентность для реалистичности
        self.position += np.array([
            random.uniform(-0.1, 0.1),
            random.uniform(-0.1, 0.1),
            random.uniform(-0.05, 0.05)
        ])

    def _update_battery(self, dt: float):
        """Обновление уровня батареи"""
        # Простая модель расхода батареи
        speed = np.linalg.norm(self.velocity)
        battery_drain = dt * (0.1 + 0.1 * speed / self.max_speed)
        
        # Дополнительный расход при атаке
        if self.state == DroneState.ATTACK:
            battery_drain *= 1.5
        
        self.battery_level = max(0.0, self.battery_level - battery_drain)
        
        # Автоматическое возвращение на базу при низком заряде
        if self.battery_level < 20.0 and self.state != DroneState.RTB:
            self.return_to_base()
            print(f"Drone {self.id} low battery ({self.battery_level:.1f}%). Returning to base.")

    def _check_boundaries(self):
        """Проверка и корректировка границ полета"""
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

    def set_target(self, target_position: np.ndarray):
        """Установка целевой позиции"""
        self.target_position = target_position
        self.state = DroneState.TARGET_TRACKING

    def move_to(self, target_position: np.ndarray, max_speed: Optional[float] = None):
        """Движение к указанной позиции"""
        if max_speed is None:
            max_speed = self.max_speed

        direction = target_position - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:  # Если расстояние существенное
            # Нормализованный вектор направления
            direction = direction / distance
            
            # Расчет желаемой скорости
            desired_velocity = direction * max_speed
            
            # Расчет ускорения для достижения желаемой скорости
            velocity_diff = desired_velocity - self.velocity
            self.acceleration = velocity_diff / 0.1  # Время реакции 0.1 секунды
            
            # Ограничение ускорения
            acc_magnitude = np.linalg.norm(self.acceleration)
            if acc_magnitude > self.max_acceleration:
                self.acceleration = (self.acceleration / acc_magnitude) * self.max_acceleration
        else:
            # Если мы близко к цели, останавливаемся
            self.velocity = np.zeros(3)
            self.acceleration = np.zeros(3)

    def start_patrol(self, waypoints: List[np.ndarray]):
        """Начало патрулирования по заданным точкам"""
        if not waypoints:
            # Если точки не предоставлены, создаем случайные вокруг текущей позиции
            waypoints = self._generate_random_patrol_points()
            
        self.patrol_waypoints = waypoints
        self.current_waypoint_index = 0
        self.state = DroneState.PATROL

    def _generate_random_patrol_points(self, num_points: int = 5, radius: float = 500.0) -> List[np.ndarray]:
        """Генерация случайных точек патрулирования вокруг текущей позиции"""
        waypoints = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            z = self.position[2] + random.uniform(-50, 50)
            
            # Обеспечиваем, что высота в допустимых пределах
            z = max(self.flight_boundaries['z_min'], min(z, self.flight_boundaries['z_max']))
            
            waypoint = self.position + np.array([dx, dy, 0])
            waypoint[2] = z
            waypoints.append(waypoint)
            
        return waypoints

    def update_patrol(self):
        """Обновление патрулирования"""
        if not self.patrol_waypoints:
            # Если нет точек патрулирования, создаем их
            self.patrol_waypoints = self._generate_random_patrol_points()

        current_waypoint = self.patrol_waypoints[self.current_waypoint_index]
        distance = np.linalg.norm(current_waypoint - self.position)

        if distance < 2.0:  # Если достигли текущей точки
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.patrol_waypoints)
            current_waypoint = self.patrol_waypoints[self.current_waypoint_index]
            print(f"Drone {self.id} reached waypoint {self.current_waypoint_index-1}. "
                  f"Distance traveled: {self.distance_traveled/1000:.2f} km")

        self.move_to(current_waypoint, max_speed=self.max_speed * 0.7)  # Патрулируем на 70% от макс. скорости

    def return_to_base(self):
        """Возвращение на базу"""
        self.state = DroneState.RTB
        self.target_position = self.home_position
        self.move_to(self.home_position)

    def attack_target(self, target_position: np.ndarray):
        """Атака цели"""
        if self.state != DroneState.ATTACK:
            self.state = DroneState.ATTACK
            self.target_position = target_position

        distance = np.linalg.norm(target_position - self.position)
        
        if distance <= self.attack_range:
            # Выполнение атаки
            # Имитация времени, необходимого для атаки
            time_to_attack = 3.0  # секунды
            
            # Возвращаем True, если атака успешна (в реальной системе будет более сложная логика)
            attack_success = random.random() < 0.7  # 70% шанс успешной атаки
            
            if attack_success:
                print(f"Drone {self.id} successfully attacked target at {target_position}!")
                
            return attack_success
        else:
            # Движение к цели
            self.move_to(target_position)
            return False

    def set_nearby_drones(self, drone_positions: List[Tuple[int, np.ndarray]]):
        """Обновление информации о ближайших дронах для предотвращения столкновений"""
        self.nearby_drones = []
        for drone_id, position in drone_positions:
            if drone_id != self.id:  # Исключаем себя
                distance = np.linalg.norm(position - self.position)
                if distance < self.collision_avoidance_radius:
                    self.nearby_drones.append((drone_id, position, distance))

    def _apply_collision_avoidance(self):
        """Применение алгоритма предотвращения столкновений"""
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
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    random.uniform(-1, 1)
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

    def get_status(self) -> dict:
        """Получение текущего статуса дрона"""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'state': self.state.value,
            'battery': self.battery_level,
            'distance_traveled': self.distance_traveled,
            'target_position': self.target_position.tolist() if self.target_position is not None else None
        }

    def emergency_stop(self):
        """Экстренная остановка"""
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.state = DroneState.RTB 