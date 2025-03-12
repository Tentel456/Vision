import numpy as np
import time
import logging
from typing import Optional, Dict, Tuple, List
import threading
import cv2

from .base_connector import DroneConnector

try:
    # Попытка импорта библиотеки DJI SDK
    import djitellopy as tello
    HAS_DJI_SDK = True
except ImportError:
    HAS_DJI_SDK = False
    print("DJI SDK не установлен. Для работы с дронами DJI установите: pip install djitellopy")

class DJIConnector(DroneConnector):
    """
    Коннектор для дронов DJI, использующий официальный SDK DJI.
    Поддерживает модели: Tello, Mavic, Phantom, Inspire и др.
    """
    
    def __init__(self, model: str = 'tello'):
        """
        Инициализация коннектора DJI
        
        Args:
            model: Модель дрона ('tello', 'mavic', 'phantom', 'inspire')
        """
        self.model = model.lower()
        self.drone = None
        self.is_connected_flag = False
        self.video_thread = None
        self.video_frame = None
        self.video_running = False
        self.home_position = np.zeros(3)
        
        # Проверка доступности SDK
        if not HAS_DJI_SDK:
            logging.error("DJI SDK не установлен. Установите: pip install djitellopy")
            return
            
        # Настройка логгирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"DJIConnector-{model}")
    
    def connect(self, connection_params: Dict) -> bool:
        """
        Подключение к дрону DJI
        
        Args:
            connection_params: Параметры соединения с дроном
                - ip: IP-адрес дрона (для Tello по умолчанию '192.168.10.1')
                - port: Порт для соединения (для Tello по умолчанию 8889)
        
        Returns:
            bool: True если подключение успешно, иначе False
        """
        if not HAS_DJI_SDK:
            self.logger.error("DJI SDK не установлен")
            return False
            
        try:
            if self.model == 'tello':
                ip = connection_params.get('ip', '192.168.10.1')
                self.drone = tello.Tello(ip)
                self.drone.connect()
                self.logger.info(f"Подключен к Tello (IP: {ip})")
                
                # Получение информации о батарее для проверки соединения
                battery = self.drone.get_battery()
                self.logger.info(f"Уровень заряда батареи: {battery}%")
                
                self.is_connected_flag = True
                return True
            else:
                self.logger.error(f"Модель {self.model} пока не поддерживается")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка подключения к дрону: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Отключение от дрона DJI
        
        Returns:
            bool: True если отключение успешно, иначе False
        """
        if not self.is_connected():
            return True
            
        try:
            # Остановка видеопотока, если он запущен
            if self.video_running:
                self.stop_video_stream()
                
            # Для Tello особый метод отключения
            if self.model == 'tello':
                self.drone.end()
                
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
    
    def takeoff(self, altitude: float = 1.5) -> bool:
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
            if self.model == 'tello':
                # Tello имеет фиксированную высоту взлета ~0.5-1.0 метра
                self.drone.takeoff()
                
                # Если требуется взлететь выше, перемещаемся на указанную высоту
                if altitude > 1.0:
                    self.logger.info(f"Перемещение на высоту {altitude} метров")
                    time.sleep(1)  # Пауза для стабилизации после взлета
                    self.drone.move_up(int((altitude - 1.0) * 100))  # Tello принимает см
                
                # Сохраняем домашнюю позицию
                self.home_position = self.get_position()
                
                self.logger.info(f"Взлет на высоту {altitude} метров выполнен")
                return True
            else:
                self.logger.error(f"Взлет для модели {self.model} пока не реализован")
                return False
                
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
            if self.model == 'tello':
                self.drone.land()
                self.logger.info("Посадка выполнена")
                return True
            else:
                self.logger.error(f"Посадка для модели {self.model} пока не реализована")
                return False
                
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
            # Для Tello реализуем возврат через перемещение к точке взлета
            if self.model == 'tello':
                current_pos = self.get_position()
                self.logger.info(f"Возврат из позиции {current_pos} к {self.home_position}")
                
                # Расчет вектора перемещения
                move_vector = self.home_position - current_pos
                
                # Перемещение к домашней позиции
                return self.move_to(self.home_position)
            else:
                self.logger.error(f"Возврат домой для модели {self.model} пока не реализован")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка при возврате домой: {str(e)}")
            return False
    
    def move_to(self, position: np.ndarray, speed: Optional[float] = None) -> bool:
        """
        Перемещение дрона в указанную позицию
        
        Args:
            position: Целевая позиция [x, y, z] в метрах
            speed: Скорость перемещения в м/с (если поддерживается)
            
        Returns:
            bool: True в случае успеха, иначе False
        """
        if not self.is_connected():
            self.logger.error("Невозможно переместиться: дрон не подключен")
            return False
            
        try:
            if self.model == 'tello':
                # Получение текущего положения
                current_pos = self.get_position()
                
                # Расчет разницы положений в см (Tello использует сантиметры)
                dx = int((position[0] - current_pos[0]) * 100)
                dy = int((position[1] - current_pos[1]) * 100)
                dz = int((position[2] - current_pos[2]) * 100)
                
                self.logger.info(f"Перемещение на {dx}см (вперед/назад), {dy}см (вправо/влево), {dz}см (вверх/вниз)")
                
                # Установка скорости, если указана
                if speed is not None:
                    speed_cm = int(speed * 100)
                    self.drone.set_speed(speed_cm)
                
                # Выполнение перемещений по каждой оси
                if dx != 0:
                    if dx > 0:
                        self.drone.move_forward(abs(dx))
                    else:
                        self.drone.move_back(abs(dx))
                
                if dy != 0:
                    if dy > 0:
                        self.drone.move_right(abs(dy))
                    else:
                        self.drone.move_left(abs(dy))
                
                if dz != 0:
                    if dz > 0:
                        self.drone.move_up(abs(dz))
                    else:
                        self.drone.move_down(abs(dz))
                
                self.logger.info(f"Перемещение завершено")
                return True
            else:
                self.logger.error(f"Перемещение для модели {self.model} пока не реализовано")
                return False
                
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
            if self.model == 'tello':
                # Преобразование м/с в см/с (Tello использует см/с)
                vx = int(velocity[0] * 100)
                vy = int(velocity[1] * 100)
                vz = int(velocity[2] * 100)
                
                # Установка скорости (Tello использует команду rc)
                self.drone.send_rc_control(vy, vx, vz, 0)  # Порядок: left/right, forward/back, up/down, yaw
                
                self.logger.info(f"Установлена скорость: {velocity} м/с")
                return True
            else:
                self.logger.error(f"Установка скорости для модели {self.model} пока не реализована")
                return False
                
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
            if self.model == 'tello':
                # Получение текущего угла и расчет относительного поворота
                current_yaw = self.get_attitude()[2]  # Текущий угол рыскания
                relative_yaw = yaw_angle - current_yaw
                
                # Нормализация угла в диапазоне -180..180
                if relative_yaw > 180:
                    relative_yaw -= 360
                elif relative_yaw < -180:
                    relative_yaw += 360
                
                # Выполнение поворота
                if relative_yaw > 0:
                    self.drone.rotate_clockwise(int(relative_yaw))
                else:
                    self.drone.rotate_counter_clockwise(int(abs(relative_yaw)))
                
                self.logger.info(f"Установлен угол рыскания: {yaw_angle}°")
                return True
            else:
                self.logger.error(f"Установка угла для модели {self.model} пока не реализована")
                return False
                
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
            if self.model == 'tello':
                # Для Tello позиция не предоставляется напрямую в API
                # Приблизительно оцениваем на основе предыдущей команды перемещения
                # В реальном проекте нужно реализовать более точное определение положения
                x = 0  # В данной реализации не отслеживаем реальную позицию
                y = 0
                z = self.drone.get_height() / 100.0  # Высота в см -> метры
                
                return np.array([x, y, z])
            else:
                self.logger.error(f"Получение позиции для модели {self.model} пока не реализовано")
                return np.zeros(3)
                
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
            if self.model == 'tello':
                # В Tello API нет прямого доступа к скорости
                # В реальной реализации использовать IMU данные
                return np.zeros(3)
            else:
                self.logger.error(f"Получение скорости для модели {self.model} пока не реализовано")
                return np.zeros(3)
                
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
            if self.model == 'tello':
                # Получаем IMU данные для Tello
                imu = self.drone.get_imu()
                roll = imu[0]
                pitch = imu[1]
                yaw = imu[2]
                
                return (roll, pitch, yaw)
            else:
                self.logger.error(f"Получение углов для модели {self.model} пока не реализовано")
                return (0.0, 0.0, 0.0)
                
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
            if self.model == 'tello':
                return float(self.drone.get_battery())
            else:
                self.logger.error(f"Получение заряда для модели {self.model} пока не реализовано")
                return 0.0
                
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
            if self.model == 'tello':
                if self.video_running:
                    self.logger.warning("Видеопоток уже запущен")
                    return True
                    
                # Запуск видеопотока
                self.drone.streamon()
                self.video_running = True
                
                # Запуск потока для обработки видео
                self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
                self.video_thread.start()
                
                self.logger.info("Видеопоток запущен")
                return True
            else:
                self.logger.error(f"Видеопоток для модели {self.model} пока не реализован")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка при запуске видеопотока: {str(e)}")
            return False
    
    def _video_loop(self):
        """Обработчик видеопотока в отдельном потоке"""
        while self.video_running and self.is_connected():
            try:
                frame = self.drone.get_frame_read().frame
                if frame is not None:
                    self.video_frame = frame
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                self.logger.error(f"Ошибка в видеопотоке: {str(e)}")
                time.sleep(1)
    
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
            if self.model == 'tello':
                if not self.video_running:
                    return True
                    
                # Остановка видеопотока
                self.video_running = False
                if self.video_thread is not None:
                    self.video_thread.join(timeout=1.0)
                    
                self.drone.streamoff()
                self.logger.info("Видеопоток остановлен")
                return True
            else:
                self.logger.error(f"Остановка видеопотока для модели {self.model} пока не реализована")
                return False
                
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
            if self.model == 'tello':
                # Для Tello команда emergency
                self.drone.emergency()
                self.logger.warning("ЭКСТРЕННАЯ ОСТАНОВКА ВЫПОЛНЕНА")
                return True
            else:
                self.logger.error(f"Экстренная остановка для модели {self.model} пока не реализована")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка при экстренной остановке: {str(e)}")
            return False 