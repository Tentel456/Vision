import numpy as np
import time
import logging
import threading
import cv2
from typing import Optional, Dict, Tuple, List

from .base_connector import DroneConnector

try:
    # Попытка импорта библиотеки для работы с Parrot (Olympe)
    import olympe
    from olympe.messages.ardrone3.Piloting import TakeOff, Landing, Emergency, PCMD
    from olympe.messages.ardrone3.PilotingState import FlyingStateChanged, PositionChanged, SpeedChanged, AttitudeChanged
    from olympe.messages.ardrone3.PilotingSettings import MaxTilt
    from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
    from olympe.messages.common.CommonState import BatteryStateChanged
    HAS_OLYMPE = True
except ImportError:
    HAS_OLYMPE = False
    print("Olympe не установлен. Для работы с дронами Parrot установите: pip install olympe-linux пакет (только для Linux)")

class ParrotConnector(DroneConnector):
    """
    Коннектор для дронов Parrot, использующий SDK Olympe.
    Поддерживает модели: Bebop, Anafi.
    """
    
    def __init__(self, model: str = 'anafi'):
        """
        Инициализация коннектора для дронов Parrot
        
        Args:
            model: Модель дрона ('anafi', 'bebop')
        """
        self.model = model.lower()
        self.drone = None
        self.is_connected_flag = False
        self.video_thread = None
        self.video_frame = None
        self.video_running = False
        self.home_position = np.zeros(3)
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_attitude = (0.0, 0.0, 0.0)
        self.battery_level = 0.0
        
        # Проверка доступности Olympe SDK
        if not HAS_OLYMPE:
            logging.error("Olympe SDK не установлен. Работает только на Linux.")
            return
            
        # Настройка логгирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"ParrotConnector-{model}")
    
    def connect(self, connection_params: Dict) -> bool:
        """
        Подключение к дрону Parrot
        
        Args:
            connection_params: Параметры соединения с дроном
                - ip: IP-адрес дрона (по умолчанию '192.168.42.1')
                - streaming_port: Порт для видеопотока (по умолчанию 55004)
        
        Returns:
            bool: True если подключение успешно, иначе False
        """
        if not HAS_OLYMPE:
            self.logger.error("Olympe SDK не установлен")
            return False
            
        try:
            # Получение параметров соединения
            ip = connection_params.get('ip', '192.168.42.1')
            streaming_port = connection_params.get('streaming_port', 55004)
            
            self.logger.info(f"Подключение к дрону Parrot {self.model.upper()} по IP: {ip}")
            
            # Создание объекта дрона
            self.drone = olympe.Drone(ip)
            
            # Подключение к дрону
            connection = self.drone.connect()
            if not connection.wait(_timeout=10):
                self.logger.error("Ошибка подключения к дрону")
                return False
            
            # Настройка получения данных о состоянии дрона
            self._setup_drone_callbacks()
            
            self.is_connected_flag = True
            self.logger.info("Подключение к дрону успешно установлено")
            
            # Получение информации о дроне
            self.battery_level = self.drone.get_state(BatteryStateChanged)["percent"]
            self.logger.info(f"Заряд батареи: {self.battery_level}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка подключения к дрону: {str(e)}")
            return False
    
    def _setup_drone_callbacks(self):
        """Настройка коллбэков для получения данных от дрона"""
        try:
            # Установка обработчиков событий для получения данных от дрона
            self.drone.subscribe(
                # Обработчики состояния полета
                FlyingStateChanged(_policy='wait') |
                PositionChanged(_policy='wait') |
                SpeedChanged(_policy='wait') |
                AttitudeChanged(_policy='wait') |
                BatteryStateChanged(_policy='wait'),
                self._drone_state_callback
            )
            
            self.logger.info("Обработчики событий установлены")
            
        except Exception as e:
            self.logger.error(f"Ошибка настройки обработчиков: {str(e)}")
    
    def _drone_state_callback(self, event, scheduler):
        """Обработчик событий от дрона"""
        # Обновление позиции
        if isinstance(event, PositionChanged):
            self.current_position = np.array([
                event.latitude,  # Широта
                event.longitude, # Долгота
                event.altitude   # Высота
            ])
            
        # Обновление скорости
        elif isinstance(event, SpeedChanged):
            self.current_velocity = np.array([
                event.speedX,  # Скорость по X
                event.speedY,  # Скорость по Y
                event.speedZ   # Скорость по Z
            ])
            
        # Обновление ориентации
        elif isinstance(event, AttitudeChanged):
            self.current_attitude = (
                event.roll,   # Крен
                event.pitch,  # Тангаж
                event.yaw     # Рыскание
            )
            
        # Обновление заряда батареи
        elif isinstance(event, BatteryStateChanged):
            self.battery_level = event.percent
    
    def disconnect(self) -> bool:
        """
        Отключение от дрона
        
        Returns:
            bool: True если отключение успешно, иначе False
        """
        if not self.is_connected():
            return True
            
        try:
            # Остановка видеопотока, если он запущен
            if self.video_running:
                self.stop_video_stream()
                
            # Отключение от дрона
            self.drone.disconnect()
            self.drone = None
            self.is_connected_flag = False
            
            self.logger.info("Отключено от дрона")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при отключении: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Проверка соединения с дроном
        
        Returns:
            bool: True если дрон подключен, иначе False
        """
        return self.is_connected_flag and self.drone is not None
    
    def takeoff(self, altitude: float) -> bool:
        """
        Взлет дрона на указанную высоту
        
        Args:
            altitude: Высота взлета в метрах
            
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно взлететь: дрон не подключен")
            return False
            
        try:
            # Команда взлета
            self.logger.info(f"Взлет дрона...")
            
            # В Olympe высота взлета фиксированная, затем можно подняться на нужную высоту
            takeoff = self.drone(TakeOff())
            if not takeoff.wait(_timeout=10):
                self.logger.error("Ошибка взлета")
                return False
                
            # Ждем, пока дрон не начнет полет
            self.drone(FlyingStateChanged(state="hovering", _policy="wait", _timeout=10))
            
            # Если нужно подняться выше стандартной высоты взлета
            if altitude > 2.0:  # Стандартная высота взлета обычно 1-2 метра
                self.logger.info(f"Подъем на высоту {altitude} метров...")
                
                # Подъем на указанную высоту
                # Для Anafi и Bebop используем PCMD для управления движением
                # Отрицательное значение по Z означает подъем
                start_time = time.time()
                while time.time() - start_time < 5.0:  # Ограничение времени подъема
                    self.drone(PCMD(
                        1,           # 1 означает активное управление
                        0,           # roll (крен)
                        0,           # pitch (тангаж)
                        0,           # yaw (рыскание)
                        -50,         # gaz (вертикальная скорость), отрицательные значения - подъем
                        0            # время для выполнения команды
                    ))
                    time.sleep(0.1)
                
                # Остановка вертикального движения
                self.drone(PCMD(1, 0, 0, 0, 0, 0))
            
            # Сохраняем домашнюю позицию
            self.home_position = self.get_position()
            
            self.logger.info("Взлет завершен")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при взлете: {str(e)}")
            return False
    
    def land(self) -> bool:
        """
        Посадка дрона
        
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно приземлиться: дрон не подключен")
            return False
            
        try:
            # Команда посадки
            self.logger.info("Посадка дрона...")
            
            landing = self.drone(Landing())
            if not landing.wait(_timeout=10):
                self.logger.error("Ошибка при посадке")
                return False
                
            # Ждем, пока дрон не приземлится
            self.drone(FlyingStateChanged(state="landed", _policy="wait", _timeout=10))
            
            self.logger.info("Посадка завершена")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при посадке: {str(e)}")
            return False
    
    def return_to_home(self) -> bool:
        """
        Возврат дрона к точке взлета
        
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно вернуться домой: дрон не подключен")
            return False
            
        try:
            # Для Parrot дронов используем перемещение к сохраненной точке взлета
            self.logger.info("Возврат к точке взлета...")
            
            # Используем move_to для перемещения к домашней позиции
            return self.move_to(self.home_position)
            
        except Exception as e:
            self.logger.error(f"Ошибка при возврате домой: {str(e)}")
            return False
    
    def move_to(self, position: np.ndarray, speed: Optional[float] = None) -> bool:
        """
        Перемещение дрона в указанную позицию
        
        Args:
            position: Целевая позиция [x, y, z] в метрах
            speed: Скорость перемещения в м/с
            
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно переместиться: дрон не подключен")
            return False
            
        try:
            # Получение текущего положения
            current_pos = self.get_position()
            
            # Расчет разницы положений
            dx = position[0] - current_pos[0]
            dy = position[1] - current_pos[1]
            dz = position[2] - current_pos[2]
            
            # Расчет направления движения
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            if distance < 0.1:  # Если уже близко к цели
                self.logger.info("Уже в целевой позиции")
                return True
                
            self.logger.info(f"Перемещение на {distance:.2f} метров")
            
            # Нормализация направления
            direction_x = dx / distance
            direction_y = dy / distance
            direction_z = dz / distance
            
            # Установка максимального наклона дрона
            max_tilt = 15  # Градусы
            self.drone(MaxTilt(max_tilt)).wait()
            
            # Определение скорости
            move_speed = 0.5  # Скорость по умолчанию (50%)
            if speed is not None:
                # Преобразование м/с в процент от максимальной скорости (примерно)
                move_speed = min(1.0, speed / 10.0)  # Предполагаем максимальную скорость 10 м/с
            
            # Расчет необходимого времени для перемещения (с запасом)
            estimated_time = distance / (move_speed * 5) + 2.0  # В секундах
            
            # Создаем команду для перемещения
            start_time = time.time()
            while time.time() - start_time < estimated_time:
                # Для Parrot используем PCMD для прямого управления
                # PCMD принимает значения от -100 до 100
                roll_cmd = int(direction_y * 100 * move_speed)
                pitch_cmd = int(direction_x * 100 * move_speed)
                gaz_cmd = int(-direction_z * 100 * move_speed)  # Отрицательное значение для подъема
                
                # Ограничение значений
                roll_cmd = max(-100, min(100, roll_cmd))
                pitch_cmd = max(-100, min(100, pitch_cmd))
                gaz_cmd = max(-100, min(100, gaz_cmd))
                
                self.drone(PCMD(
                    1,           # 1 означает активное управление
                    roll_cmd,    # roll (крен)
                    pitch_cmd,   # pitch (тангаж)
                    0,           # yaw (рыскание)
                    gaz_cmd,     # gaz (вертикальная скорость)
                    0            # время для выполнения команды
                ))
                time.sleep(0.1)
            
            # Остановка движения
            self.drone(PCMD(1, 0, 0, 0, 0, 0))
            
            self.logger.info("Перемещение завершено")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при перемещении: {str(e)}")
            return False
    
    def set_velocity(self, velocity: np.ndarray) -> bool:
        """
        Установка скорости дрона
        
        Args:
            velocity: Вектор скорости [vx, vy, vz] в м/с
            
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно установить скорость: дрон не подключен")
            return False
            
        try:
            # Преобразование м/с в проценты для PCMD (-100 до 100)
            # Предполагаем максимальную скорость 10 м/с
            max_speed = 10.0
            roll_cmd = int((velocity[1] / max_speed) * 100)
            pitch_cmd = int((velocity[0] / max_speed) * 100)
            gaz_cmd = int((-velocity[2] / max_speed) * 100)  # Отрицательное значение для подъема
            
            # Ограничение значений
            roll_cmd = max(-100, min(100, roll_cmd))
            pitch_cmd = max(-100, min(100, pitch_cmd))
            gaz_cmd = max(-100, min(100, gaz_cmd))
            
            # Установка скорости
            self.drone(PCMD(
                1,           # 1 означает активное управление
                roll_cmd,    # roll (крен)
                pitch_cmd,   # pitch (тангаж)
                0,           # yaw (рыскание)
                gaz_cmd,     # gaz (вертикальная скорость)
                0            # время для выполнения команды
            ))
            
            self.logger.info(f"Установлена скорость: {velocity} м/с")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при установке скорости: {str(e)}")
            return False
    
    def set_yaw(self, yaw_angle: float) -> bool:
        """
        Установка угла рыскания дрона
        
        Args:
            yaw_angle: Угол в градусах (0-360)
            
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно установить угол: дрон не подключен")
            return False
            
        try:
            # Получение текущего угла и расчет относительного поворота
            current_yaw = self.get_attitude()[2]  # Текущий угол рыскания
            relative_yaw = yaw_angle - current_yaw
            
            # Нормализация угла в диапазоне -180..180
            if relative_yaw > 180:
                relative_yaw -= 360
            elif relative_yaw < -180:
                relative_yaw += 360
                
            # Направление поворота и скорость (в процентах от максимальной)
            yaw_speed = min(100, abs(relative_yaw) / 1.8)  # Максимальная скорость при повороте на 180
            yaw_cmd = int(yaw_speed) if relative_yaw > 0 else -int(yaw_speed)
            
            # Выполнение поворота
            start_time = time.time()
            estimated_time = abs(relative_yaw) / 90.0 + 1.0  # Примерное время для поворота
            
            while time.time() - start_time < estimated_time:
                self.drone(PCMD(
                    1,           # 1 означает активное управление
                    0,           # roll (крен)
                    0,           # pitch (тангаж)
                    yaw_cmd,     # yaw (рыскание)
                    0,           # gaz (вертикальная скорость)
                    0            # время для выполнения команды
                ))
                time.sleep(0.1)
            
            # Остановка поворота
            self.drone(PCMD(1, 0, 0, 0, 0, 0))
            
            self.logger.info(f"Установлен угол рыскания: {yaw_angle}°")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при установке угла: {str(e)}")
            return False
    
    def get_position(self) -> np.ndarray:
        """
        Получение текущей позиции дрона
        
        Returns:
            np.ndarray: Позиция [x, y, z] в метрах
        """
        if not self.is_connected():
            self.logger.error("Невозможно получить позицию: дрон не подключен")
            return np.zeros(3)
            
        try:
            # Возвращаем последнюю известную позицию
            # В реальном проекте нужно преобразовать GPS координаты в метры
            return self.current_position
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении позиции: {str(e)}")
            return np.zeros(3)
    
    def get_velocity(self) -> np.ndarray:
        """
        Получение текущей скорости дрона
        
        Returns:
            np.ndarray: Вектор скорости [vx, vy, vz] в м/с
        """
        if not self.is_connected():
            self.logger.error("Невозможно получить скорость: дрон не подключен")
            return np.zeros(3)
            
        try:
            # Возвращаем последнюю известную скорость
            return self.current_velocity
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении скорости: {str(e)}")
            return np.zeros(3)
    
    def get_attitude(self) -> Tuple[float, float, float]:
        """
        Получение углов ориентации дрона
        
        Returns:
            Tuple[float, float, float]: Крен, тангаж, рыскание в градусах
        """
        if not self.is_connected():
            self.logger.error("Невозможно получить углы: дрон не подключен")
            return (0.0, 0.0, 0.0)
            
        try:
            # Преобразование радианов в градусы
            roll = np.degrees(self.current_attitude[0])
            pitch = np.degrees(self.current_attitude[1])
            yaw = np.degrees(self.current_attitude[2])
            
            return (roll, pitch, yaw)
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении углов: {str(e)}")
            return (0.0, 0.0, 0.0)
    
    def get_battery_level(self) -> float:
        """
        Получение уровня заряда батареи
        
        Returns:
            float: Уровень заряда в процентах (0-100)
        """
        if not self.is_connected():
            self.logger.error("Невозможно получить заряд батареи: дрон не подключен")
            return 0.0
            
        try:
            return self.battery_level
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении заряда батареи: {str(e)}")
            return 0.0
    
    def start_video_stream(self, resolution: Tuple[int, int] = (1280, 720)) -> bool:
        """
        Запуск видеопотока с камеры дрона
        
        Args:
            resolution: Разрешение видео (ширина, высота)
            
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно запустить видеопоток: дрон не подключен")
            return False
            
        try:
            if self.video_running:
                self.logger.warning("Видеопоток уже запущен")
                return True
                
            # Запуск видеопотока
            self.drone.set_streaming_output_files()
            self.drone.start_video_streaming()
            
            # Запуск потока для обработки видео
            self.video_running = True
            self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.video_thread.start()
            
            self.logger.info("Видеопоток запущен")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при запуске видеопотока: {str(e)}")
            return False
    
    def _video_loop(self):
        """Обработчик видеопотока в отдельном потоке"""
        try:
            while self.video_running and self.is_connected():
                # В реальном проекте здесь будет код для получения видеопотока 
                # с использованием Olympe
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Пустой кадр для примера
                
                self.video_frame = frame
                time.sleep(0.03)  # ~30 FPS
                
        except Exception as e:
            self.logger.error(f"Ошибка в видеопотоке: {str(e)}")
            self.video_running = False
    
    def stop_video_stream(self) -> bool:
        """
        Остановка видеопотока
        
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно остановить видеопоток: дрон не подключен")
            return False
            
        try:
            if not self.video_running:
                return True
                
            # Остановка видеопотока
            self.video_running = False
            if self.video_thread is not None:
                self.video_thread.join(timeout=1.0)
                
            self.drone.stop_video_streaming()
            
            self.logger.info("Видеопоток остановлен")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при остановке видеопотока: {str(e)}")
            return False
    
    def get_video_frame(self) -> Optional[np.ndarray]:
        """
        Получение текущего кадра с камеры
        
        Returns:
            Optional[np.ndarray]: Кадр в формате OpenCV (None если видеопоток не запущен)
        """
        if not self.video_running or self.video_frame is None:
            return None
            
        return self.video_frame.copy()
    
    def emergency_stop(self) -> bool:
        """
        Экстренная остановка дрона
        
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно выполнить экстренную остановку: дрон не подключен")
            return False
            
        try:
            # Команда экстренной остановки
            emergency = self.drone(Emergency())
            if not emergency.wait(_timeout=5):
                self.logger.error("Ошибка при экстренной остановке")
                return False
                
            self.logger.warning("ЭКСТРЕННАЯ ОСТАНОВКА ВЫПОЛНЕНА")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при экстренной остановке: {str(e)}")
            return False 