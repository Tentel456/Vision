import numpy as np
from typing import List, Dict, Tuple
from enum import Enum
import threading
import time
import random

class DroneState(Enum):
    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    PATROL = "PATROL"
    TARGET_TRACKING = "TARGET_TRACKING"
    ATTACK = "ATTACK"
    RTB = "RTB"  # Return to base

class SwarmController:
    def __init__(self, num_drones: int = 10):
        self.num_drones = num_drones
        self.drones: Dict[int, Dict] = {}
        self.targets: Dict[int, Dict] = {}
        self.attack_mode = False
        self.lock = threading.Lock()
        
        # Параметры безопасности
        self.min_safe_distance = 30.0  # минимальное безопасное расстояние между дронами (метры)
        self.patrol_area_size = 2000.0  # размер области патрулирования (метры)
        self.attack_distance_threshold = 5000.0  # пороговое расстояние для перехода в режим атаки (метры)
        
        # Initialize drone fleet
        for i in range(num_drones):
            # Распределяем дроны в разных начальных позициях для избежания столкновений
            initial_position = np.array([
                random.uniform(-50, 50),  # X
                random.uniform(-50, 50),  # Y
                random.uniform(30, 50)    # Z (высота)
            ])
            
            self.drones[i] = {
                'id': i,
                'state': DroneState.IDLE,
                'position': initial_position,
                'velocity': np.zeros(3),
                'target_id': None,
                'battery': 100,
                'active': True,
                'distance_traveled': 0.0,
                'previous_position': initial_position.copy(),
                'patrol_waypoints': [],
                'patrol_index': 0
            }

    def start_mission(self):
        """Запуск миссии роя дронов"""
        print(f"Starting mission with {self.num_drones} drones")
        
        # Генерация маршрутов патрулирования для всех дронов
        self._generate_patrol_routes()
        
        for drone_id in self.drones:
            self._launch_drone(drone_id)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_swarm)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _generate_patrol_routes(self):
        """Генерация маршрутов патрулирования для всех дронов"""
        # Разделяем область патрулирования между дронами
        sector_angle = 2 * np.pi / self.num_drones
        
        for drone_id, drone in self.drones.items():
            base_angle = drone_id * sector_angle
            patrol_waypoints = []
            
            # Генерируем 5 точек патрулирования для каждого дрона
            for i in range(5):
                # Случайное расстояние в пределах сектора дрона
                distance = random.uniform(500, self.patrol_area_size)
                # Случайный угол в пределах сектора дрона
                angle = base_angle + random.uniform(-0.2, 0.2) * sector_angle
                
                # Вычисляем координаты точки патрулирования
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)
                z = random.uniform(50, 100)  # Высота патрулирования
                
                patrol_waypoints.append(np.array([x, y, z]))
            
            # Сохраняем маршрут патрулирования для дрона
            drone['patrol_waypoints'] = patrol_waypoints
            drone['patrol_index'] = 0

    def _launch_drone(self, drone_id: int):
        """Запуск отдельного дрона"""
        self.drones[drone_id]['state'] = DroneState.TAKEOFF
        # Здесь будет код для физического запуска дрона
        print(f"Launching drone {drone_id}")
        
        # После взлета переходим в режим патрулирования
        self.drones[drone_id]['state'] = DroneState.PATROL

    def _monitor_swarm(self):
        """Мониторинг состояния роя"""
        while True:
            with self.lock:
                self._update_swarm_state()
                self._check_collision_risk()
                self._update_distance_traveled()
                self._check_attack_distance_threshold()
            time.sleep(0.1)  # 10Hz update rate

    def _update_swarm_state(self):
        """Обновление состояния роя"""
        for drone_id, drone in self.drones.items():
            if not drone['active']:
                continue
            
            if self.attack_mode:
                self._process_attack_mode(drone_id)
            else:
                self._process_patrol_mode(drone_id)

    def _update_distance_traveled(self):
        """Обновление пройденного расстояния для всех дронов"""
        for drone_id, drone in self.drones.items():
            if not drone['active']:
                continue
            
            # Вычисляем расстояние от предыдущей позиции
            distance = np.linalg.norm(drone['position'] - drone['previous_position'])
            drone['distance_traveled'] += distance
            
            # Обновляем предыдущую позицию
            drone['previous_position'] = drone['position'].copy()

    def _check_attack_distance_threshold(self):
        """Проверка достижения порогового расстояния для перехода в режим атаки"""
        if self.attack_mode:
            return
            
        # Проверяем, прошли ли все дроны необходимое расстояние
        all_drones_ready = True
        for drone in self.drones.values():
            if drone['active'] and drone['distance_traveled'] < self.attack_distance_threshold:
                all_drones_ready = False
                break
        
        if all_drones_ready:
            print(f"\nAlert: All drones have traveled {self.attack_distance_threshold/1000:.1f} km")
            print("Ready for attack mode. Type 'attack' to enable.")

    def _process_attack_mode(self, drone_id: int):
        """Обработка режима атаки для дрона"""
        drone = self.drones[drone_id]
        
        if drone['target_id'] is None:
            # Поиск новой цели
            target_id = self._find_best_target(drone_id)
            if target_id is not None:
                self._assign_target(drone_id, target_id)
        else:
            # Обновление статуса атаки цели
            target = self.targets[drone['target_id']]
            
            # Получаем расстояние до цели
            distance = np.linalg.norm(drone['position'] - target['position'])
            
            # Если дрон достиг цели, помечаем её как атакованную
            if distance < 20.0 and drone['state'] == DroneState.TARGET_TRACKING:
                drone['state'] = DroneState.ATTACK
                print(f"Drone {drone_id} is attacking target {drone['target_id']}")
                
                # После атаки освобождаем дрон для поиска новой цели
                if random.random() < 0.1:  # Имитация завершения атаки (10% шанс каждое обновление)
                    print(f"Drone {drone_id} has successfully attacked target {drone['target_id']}")
                    target['attacked'] = True
                    drone['target_id'] = None
                    drone['state'] = DroneState.PATROL

    def _process_patrol_mode(self, drone_id: int):
        """Обработка режима патрулирования"""
        drone = self.drones[drone_id]
        if drone['state'] == DroneState.PATROL:
            # Обновление позиции патрулирования
            self._update_patrol_position(drone_id)

    def _update_patrol_position(self, drone_id: int):
        """Обновление позиции патрулирования"""
        drone = self.drones[drone_id]
        
        # Убеждаемся, что у дрона есть маршрут патрулирования
        if not drone['patrol_waypoints']:
            return
            
        # Получаем текущую целевую точку
        current_waypoint = drone['patrol_waypoints'][drone['patrol_index']]
        
        # Вычисляем направление к точке
        direction = current_waypoint - drone['position']
        distance = np.linalg.norm(direction)
        
        # Если дрон достиг текущей точки, переходим к следующей
        if distance < 20.0:
            drone['patrol_index'] = (drone['patrol_index'] + 1) % len(drone['patrol_waypoints'])
            print(f"Drone {drone_id} reached waypoint, moving to next point. Distance traveled: {drone['distance_traveled']/1000:.2f} km")
        else:
            # Иначе, продолжаем двигаться к текущей точке
            if distance > 0:
                direction = direction / distance  # Нормализация вектора
                
                # Скорость пропорциональна расстоянию, но не более 15 м/с
                speed = min(distance * 0.1, 15.0)
                drone['velocity'] = direction * speed

    def _check_collision_risk(self):
        """Проверка и предотвращение столкновений между дронами"""
        for i in range(self.num_drones):
            if not self.drones[i]['active']:
                continue
                
            drone1 = self.drones[i]
            
            for j in range(i + 1, self.num_drones):
                if not self.drones[j]['active']:
                    continue
                    
                drone2 = self.drones[j]
                
                # Вычисляем расстояние между дронами
                distance = np.linalg.norm(drone1['position'] - drone2['position'])
                
                # Если дроны слишком близко друг к другу
                if distance < self.min_safe_distance:
                    # Вычисляем вектор отталкивания
                    repulsion = (drone1['position'] - drone2['position'])
                    
                    if distance > 0:
                        repulsion = repulsion / distance * (self.min_safe_distance - distance)
                    else:
                        # Если дроны в одной точке, добавляем случайное смещение
                        repulsion = np.array([
                            random.uniform(-1, 1),
                            random.uniform(-1, 1),
                            random.uniform(-1, 1)
                        ])
                    
                    # Применяем отталкивание к обоим дронам
                    drone1['velocity'] += repulsion * 0.5
                    drone2['velocity'] -= repulsion * 0.5
                    
                    # Логирование коллизии
                    if distance < self.min_safe_distance * 0.5:
                        print(f"Warning: Close proximity detected between Drone {i} and Drone {j}: {distance:.2f}m")

    def _find_best_target(self, drone_id: int) -> int:
        """Поиск оптимальной цели для дрона"""
        best_target = None
        min_distance = float('inf')
        
        drone_pos = self.drones[drone_id]['position']
        
        for target_id, target in self.targets.items():
            # Пропускаем атакованные и назначенные цели
            if target.get('attacked', False) or target.get('assigned_drone') is not None:
                continue
                
            distance = np.linalg.norm(drone_pos - target['position'])
            
            # Учитываем приоритет цели при выборе
            priority_factor = target.get('priority', 0.5)
            effective_distance = distance / priority_factor
            
            if effective_distance < min_distance:
                min_distance = effective_distance
                best_target = target_id
                
        return best_target

    def _assign_target(self, drone_id: int, target_id: int):
        """Назначение цели дрону"""
        self.drones[drone_id]['target_id'] = target_id
        self.targets[target_id]['assigned_drone'] = drone_id
        self.drones[drone_id]['state'] = DroneState.TARGET_TRACKING
        
        print(f"Drone {drone_id} assigned to target {target_id} ({self.targets[target_id]['type']})")
        
        # Обновление скорости для движения к цели
        direction = self.targets[target_id]['position'] - self.drones[drone_id]['position']
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            self.drones[drone_id]['velocity'] = direction * 15.0  # Максимальная скорость

    def set_attack_mode(self, enabled: bool):
        """Включение/выключение режима атаки"""
        with self.lock:
            self.attack_mode = enabled
            print(f"Attack mode {'enabled' if enabled else 'disabled'}")
            
            if enabled:
                for drone in self.drones.values():
                    if drone['active']:
                        drone['state'] = DroneState.TARGET_TRACKING
            else:
                for drone in self.drones.values():
                    if drone['active']:
                        drone['state'] = DroneState.PATROL

    def add_target(self, target_id: int, position: np.ndarray, target_type: str):
        """Добавление новой цели"""
        with self.lock:
            # Проверяем, существует ли уже цель с таким ID
            if target_id in self.targets:
                return
                
            self.targets[target_id] = {
                'id': target_id,
                'position': position,
                'type': target_type,
                'assigned_drone': None,
                'attacked': False,
                'priority': self._calculate_target_priority(target_type)
            }

    def _calculate_target_priority(self, target_type: str) -> float:
        """Расчет приоритета цели"""
        priority_map = {
            'tank': 1.0,
            'truck': 0.8,
            'car': 0.6,
            'person': 0.4
        }
        return priority_map.get(target_type, 0.5)

    def update_drone_position(self, drone_id: int, position: np.ndarray, velocity: np.ndarray):
        """Обновление позиции дрона"""
        with self.lock:
            if drone_id in self.drones:
                old_position = self.drones[drone_id]['position'].copy()
                self.drones[drone_id]['position'] = position
                self.drones[drone_id]['velocity'] = velocity
                
                # Обновление пройденного расстояния
                distance = np.linalg.norm(position - old_position)
                self.drones[drone_id]['distance_traveled'] += distance

    def get_swarm_status(self) -> Dict:
        """Получение статуса всего роя"""
        with self.lock:
            # Вычисляем среднее пройденное расстояние
            total_distance = sum(d['distance_traveled'] for d in self.drones.values() if d['active'])
            avg_distance = total_distance / max(1, sum(1 for d in self.drones.values() if d['active']))
            
            return {
                'num_drones': self.num_drones,
                'active_drones': sum(1 for d in self.drones.values() if d['active']),
                'attack_mode': self.attack_mode,
                'num_targets': len(self.targets),
                'assigned_targets': sum(1 for t in self.targets.values() if t['assigned_drone'] is not None),
                'attacked_targets': sum(1 for t in self.targets.values() if t.get('attacked', False)),
                'avg_distance_km': avg_distance / 1000,  # Конвертация в километры
                'min_drone_distance': self._calculate_min_drone_distance()
            }

    def _calculate_min_drone_distance(self) -> float:
        """Расчет минимального расстояния между дронами"""
        min_distance = float('inf')
        
        for i in range(self.num_drones):
            if not self.drones[i]['active']:
                continue
                
            for j in range(i + 1, self.num_drones):
                if not self.drones[j]['active']:
                    continue
                    
                distance = np.linalg.norm(self.drones[i]['position'] - self.drones[j]['position'])
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else -1

    def emergency_stop(self):
        """Экстренная остановка всех дронов"""
        with self.lock:
            print("EMERGENCY STOP INITIATED")
            for drone in self.drones.values():
                drone['state'] = DroneState.RTB
                drone['target_id'] = None
                drone['velocity'] = np.zeros(3)  # Остановка движения
            
            # Очистка назначенных целей
            for target in self.targets.values():
                target['assigned_drone'] = None
            
            self.attack_mode = False 