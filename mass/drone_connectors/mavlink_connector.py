import numpy as np
import logging
import time
import threading
import cv2
from typing import Optional, Dict, Tuple, List

from .base_connector import DroneConnector

try:
    # Попытка импорта библиотеки для работы с MAVLink
    import dronekit
    from pymavlink import mavutil
    HAS_DRONEKIT = True
except ImportError:
    HAS_DRONEKIT = False
    print("DroneKit не установлен. Для работы с дронами через MAVLink установите: pip install dronekit")

class MAVLinkConnector(DroneConnector):
    """
    Коннектор для дронов, использующих протокол MAVLink.
    Поддерживает платформы Ardupilot, PX4 и другие, совместимые с MAVLink.
    """
    
    def __init__(self, autopilot: str = 'ardupilot'):
        """
        Инициализация коннектора для MAVLink дронов
        
        Args:
            autopilot: Используемый автопилот ('ardupilot', 'px4')
        """
        self.autopilot = autopilot.lower()
        self.vehicle = None
        self.is_connected_flag = False
        self.video_thread = None
        self.video_frame = None
        self.video_running = False
        self.video_source = None
        self.home_location = None
        
        # Проверка доступности DroneKit
        if not HAS_DRONEKIT:
            logging.error("DroneKit не установлен. Установите: pip install dronekit")
            return
            
        # Настройка логгирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"MAVLinkConnector-{autopilot}")
    
    def connect(self, connection_params: Dict) -> bool:
        """
        Подключение к дрону по MAVLink
        
        Args:
            connection_params: Словарь с параметрами соединения
                - connection_string: Строка подключения (например, 'udp:127.0.0.1:14550')
                - baud: Скорость соединения для последовательного порта (по умолчанию 57600)
                - timeout: Время ожидания подключения в секундах (по умолчанию 30)
                - video_source: Источник видеопотока (для подключения к камере)
        
        Returns:
            bool: True если подключение успешно, иначе False
        """
        if not HAS_DRONEKIT:
            self.logger.error("DroneKit не установлен")
            return False
            
        try:
            # Получение параметров соединения
            connection_string = connection_params.get('connection_string')
            baud = connection_params.get('baud', 57600)
            timeout = connection_params.get('timeout', 30)
            
            if not connection_string:
                self.logger.error("connection_string не указан")
                return False
                
            self.logger.info(f"Подключение к дрону через {connection_string}")
            
            # Подключение к дрону
            self.vehicle = dronekit.connect(
                connection_string,
                wait_ready=True,
                baud=baud,
                timeout=timeout
            )
            
            if self.vehicle:
                self.is_connected_flag = True
                
                # Сохраняем домашнюю позицию
                self.home_location = self.vehicle.home_location
                
                # Настройка видеоисточника, если указан
                if 'video_source' in connection_params:
                    self.video_source = connection_params['video_source']
                
                # Вывод информации о подключенном дроне
                self.logger.info(f"Успешное подключение к дрону.")
                self.logger.info(f"Версия автопилота: {self.vehicle.version}")
                self.logger.info(f"Заряд батареи: {self.vehicle.battery.level}%")
                self.logger.info(f"GPS: {self.vehicle.gps_0}")
                
                return True
            else:
                self.logger.error("Не удалось подключиться к дрону")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка подключения к дрону: {str(e)}")
            return False
    
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
                
            # Закрытие соединения с дроном
            self.vehicle.close()
            self.vehicle = None
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
        return self.is_connected_flag and self.vehicle is not None
    
    def takeoff(self, altitude: float) -> bool:
        """
        Команда взлета на указанную высоту
        
        Args:
            altitude: Высота взлета в метрах
            
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно взлететь: дрон не подключен")
            return False
            
        try:
            # Проверяем готовность дрона к взлету
            self.logger.info("Проверка готовности к взлету...")
            
            # Ожидаем получение GPS сигнала
            while not self.vehicle.is_armable:
                self.logger.info("Ожидание готовности GPS...")
                time.sleep(1)
            
            # Установка режима GUIDED
            self.vehicle.mode = dronekit.VehicleMode("GUIDED")
            
            # Ожидаем переключения режима
            while self.vehicle.mode != "GUIDED":
                self.logger.info("Ожидание переключения в режим GUIDED...")
                time.sleep(1)
            
            # Arming дрона
            self.vehicle.armed = True
            
            # Ожидаем поднятия двигателей
            while not self.vehicle.armed:
                self.logger.info("Ожидание запуска двигателей...")
                time.sleep(1)
            
            # Команда взлета
            self.logger.info(f"Взлет на высоту {altitude} метров...")
            self.vehicle.simple_takeoff(altitude)
            
            # Ожидаем достижения заданной высоты
            while True:
                current_altitude = self.vehicle.location.global_relative_frame.alt
                self.logger.info(f"Текущая высота: {current_altitude} м")
                
                # Если достигли ~95% от целевой высоты, считаем взлет завершенным
                if current_altitude >= altitude * 0.95:
                    self.logger.info("Взлет завершен")
                    break
                    
                time.sleep(1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при взлете: {str(e)}")
            return False
    
    def land(self) -> bool:
        """
        Команда посадки
        
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно приземлиться: дрон не подключен")
            return False
            
        try:
            # Установка режима LAND
            self.vehicle.mode = dronekit.VehicleMode("LAND")
            
            self.logger.info("Начинается посадка...")
            
            # Ожидаем снижения высоты до ~0.5м
            while self.vehicle.location.global_relative_frame.alt > 0.5:
                current_altitude = self.vehicle.location.global_relative_frame.alt
                self.logger.info(f"Высота при посадке: {current_altitude} м")
                time.sleep(1)
            
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
            # Установка режима RTL (Return To Launch)
            self.vehicle.mode = dronekit.VehicleMode("RTL")
            
            self.logger.info("Возврат к точке взлета...")
            
            # Проверяем режим полета
            while self.vehicle.mode != "RTL":
                self.logger.info("Ожидание переключения в режим RTL...")
                time.sleep(1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при возврате домой: {str(e)}")
            return False
    
    def move_to(self, position: np.ndarray, speed: Optional[float] = None) -> bool:
        """
        Перемещение дрона в указанную позицию
        
        Args:
            position: Целевая позиция [x, y, z] в метрах относительно точки старта
            speed: Скорость перемещения в м/с
            
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно переместиться: дрон не подключен")
            return False
            
        try:
            # Переводим локальные координаты в глобальные
            # Это упрощенный пример, в реальности нужно использовать точные преобразования
            # с учетом начальной позиции и ориентации дрона
            
            # Получение текущего местоположения
            current_location = self.vehicle.location.global_relative_frame
            
            # Конвертация локальных смещений (в метрах) в изменения в координатах
            # Приблизительно 111111 метров на градус широты
            # Долгота зависит от широты
            earth_radius = 6378137.0  # Радиус Земли в метрах
            lat_diff = position[0] / earth_radius * 180.0 / np.pi
            lon_diff = position[1] / (earth_radius * np.cos(np.radians(current_location.lat))) * 180.0 / np.pi
            
            # Создание новой координаты
            target_location = dronekit.LocationGlobalRelative(
                current_location.lat + lat_diff,
                current_location.lon + lon_diff,
                position[2]  # Высота в метрах относительно точки взлета
            )
            
            # Команда на перемещение
            self.logger.info(f"Перемещение к точке {position}...")
            
            # Установка скорости, если указана
            if speed is not None:
                msg = self.vehicle.message_factory.command_long_encode(
                    0, 0,  # target_system, target_component
                    mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,  # command
                    0,  # confirmation
                    1,  # param1 (ground speed)
                    speed,  # param2 (скорость в м/с)
                    0,  # param3 (ускорение, не используется)
                    0,  # param4 (не используется)
                    0, 0, 0  # param5-7 (не используются)
                )
                self.vehicle.send_mavlink(msg)
            
            # Команда перемещения
            self.vehicle.simple_goto(target_location)
            
            # Ожидаем достижения позиции
            # Упрощенный вариант ожидания (в реальном проекте лучше реализовать асинхронно)
            time.sleep(5)  # Даем время на начало движения
            
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
            # Создание MAVLink сообщения для установки скорости
            msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0,  # target_system, target_component
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
                0b0000111111000111,  # type_mask (только скорости)
                0, 0, 0,  # x, y, z позиция (не используется)
                velocity[0], velocity[1], velocity[2],  # vx, vy, vz в м/с
                0, 0, 0,  # ax, ay, az ускорение (не используется)
                0, 0  # yaw, yaw_rate (не используется)
            )
            
            # Отправка сообщения
            self.vehicle.send_mavlink(msg)
            
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
            # Преобразование в радианы
            yaw_rad = np.radians(yaw_angle)
            
            # Создание MAVLink сообщения для установки угла рыскания
            msg = self.vehicle.message_factory.command_long_encode(
                0, 0,  # target_system, target_component
                mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
                0,  # confirmation
                yaw_angle,  # param1 (угол в градусах)
                0,  # param2 (скорость поворота, 0 = автоматически)
                1,  # param3 (направление, 1 = по часовой)
                0,  # param4 (0 = абсолютный угол)
                0, 0, 0  # param5-7 (не используются)
            )
            
            # Отправка сообщения
            self.vehicle.send_mavlink(msg)
            
            self.logger.info(f"Установлен угол рыскания: {yaw_angle}°")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при установке угла: {str(e)}")
            return False
    
    def get_position(self) -> np.ndarray:
        """
        Получение текущей позиции дрона
        
        Returns:
            np.ndarray: Позиция [x, y, z] в метрах относительно точки старта
        """
        if not self.is_connected():
            self.logger.error("Невозможно получить позицию: дрон не подключен")
            return np.zeros(3)
            
        try:
            # Получение текущего положения
            location = self.vehicle.location.global_relative_frame
            
            # Преобразование GPS координат в метры от точки взлета
            # Это упрощенный пример, в реальности нужно использовать точные преобразования
            
            # Если нет домашней позиции, возвращаем просто текущую высоту
            if self.home_location is None:
                return np.array([0.0, 0.0, location.alt])
            
            # Расчёт расстояния от домашней точки до текущей позиции
            earth_radius = 6378137.0  # Радиус Земли в метрах
            
            # Разница в координатах
            lat_diff = location.lat - self.home_location.lat
            lon_diff = location.lon - self.home_location.lon
            
            # Преобразование в метры
            x = lat_diff * np.pi / 180.0 * earth_radius
            y = lon_diff * np.pi / 180.0 * earth_radius * np.cos(np.radians(location.lat))
            z = location.alt
            
            return np.array([x, y, z])
            
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
            # Получение скорости из DroneKit
            velocity = self.vehicle.velocity
            
            return np.array([velocity[0], velocity[1], velocity[2]])
            
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
            # Получение углов ориентации
            attitude = self.vehicle.attitude
            
            # Преобразование радианов в градусы
            roll = np.degrees(attitude.roll)
            pitch = np.degrees(attitude.pitch)
            yaw = np.degrees(attitude.yaw)
            
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
            # Получение уровня заряда батареи
            return self.vehicle.battery.level
            
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
        if self.video_running:
            self.logger.warning("Видеопоток уже запущен")
            return True
            
        try:
            # Проверка наличия источника видео
            if self.video_source is None:
                self.logger.error("Источник видео не настроен")
                return False
                
            # Запуск потока для получения видео
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
            # Открытие источника видео
            cap = cv2.VideoCapture(self.video_source)
            
            while self.video_running:
                ret, frame = cap.read()
                if ret:
                    self.video_frame = frame
                time.sleep(0.03)  # ~30 FPS
                
            # Закрытие видеопотока
            cap.release()
            
        except Exception as e:
            self.logger.error(f"Ошибка в видеопотоке: {str(e)}")
            self.video_running = False
    
    def stop_video_stream(self) -> bool:
        """
        Остановка видеопотока
        
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.video_running:
            return True
            
        try:
            # Остановка видеопотока
            self.video_running = False
            if self.video_thread is not None:
                self.video_thread.join(timeout=1.0)
                
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
            # Disarm дрона для экстренной остановки
            self.vehicle.armed = False
            
            # Дублирование команды disarm через mavlink
            msg = self.vehicle.message_factory.command_long_encode(
                0, 0,  # target_system, target_component
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,  # command
                0,  # confirmation
                0,  # param1 (0 = disarm)
                21196,  # param2 (магическое число для force disarm)
                0, 0, 0, 0, 0  # param3-7 (не используются)
            )
            
            # Отправка сообщения
            self.vehicle.send_mavlink(msg)
            
            self.logger.warning("ЭКСТРЕННАЯ ОСТАНОВКА ВЫПОЛНЕНА")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при экстренной остановке: {str(e)}")
            return False 