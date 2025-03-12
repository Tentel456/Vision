from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Dict, Tuple

class DroneConnector(ABC):
    """
    Абстрактный базовый класс для всех коннекторов дронов.
    Определяет общий интерфейс для работы с различными типами дронов.
    """
    
    @abstractmethod
    def connect(self, connection_params: Dict) -> bool:
        """
        Установка соединения с дроном
        
        Args:
            connection_params: Параметры соединения с дроном
            
        Returns:
            bool: True если соединение успешно установлено, иначе False
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Закрытие соединения с дроном
        
        Returns:
            bool: True если соединение успешно закрыто, иначе False
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Проверка активности соединения
        
        Returns:
            bool: True если соединение активно, иначе False
        """
        pass
    
    @abstractmethod
    def takeoff(self, altitude: float) -> bool:
        """
        Команда взлета на заданную высоту
        
        Args:
            altitude: Высота взлета в метрах
            
        Returns:
            bool: True если команда взлета успешно отправлена, иначе False
        """
        pass
    
    @abstractmethod
    def land(self) -> bool:
        """
        Команда посадки
        
        Returns:
            bool: True если команда посадки успешно отправлена, иначе False
        """
        pass
    
    @abstractmethod
    def return_to_home(self) -> bool:
        """
        Команда возврата на точку взлета
        
        Returns:
            bool: True если команда возврата успешно отправлена, иначе False
        """
        pass
    
    @abstractmethod
    def move_to(self, position: np.ndarray, speed: Optional[float] = None) -> bool:
        """
        Команда перемещения дрона в заданную позицию
        
        Args:
            position: Позиция (x, y, z) в метрах относительно точки старта
            speed: Скорость перемещения в м/с (опционально)
            
        Returns:
            bool: True если команда перемещения успешно отправлена, иначе False
        """
        pass
    
    @abstractmethod
    def set_velocity(self, velocity: np.ndarray) -> bool:
        """
        Установка скорости дрона
        
        Args:
            velocity: Вектор скорости (vx, vy, vz) в м/с
            
        Returns:
            bool: True если команда успешно отправлена, иначе False
        """
        pass
    
    @abstractmethod
    def set_yaw(self, yaw_angle: float) -> bool:
        """
        Установка угла рыскания (поворот вокруг вертикальной оси)
        
        Args:
            yaw_angle: Угол в градусах (0-360)
            
        Returns:
            bool: True если команда успешно отправлена, иначе False
        """
        pass
    
    @abstractmethod
    def get_position(self) -> np.ndarray:
        """
        Получение текущей позиции дрона
        
        Returns:
            np.ndarray: Позиция (x, y, z) в метрах относительно точки старта
        """
        pass
    
    @abstractmethod
    def get_velocity(self) -> np.ndarray:
        """
        Получение текущей скорости дрона
        
        Returns:
            np.ndarray: Вектор скорости (vx, vy, vz) в м/с
        """
        pass
    
    @abstractmethod
    def get_attitude(self) -> Tuple[float, float, float]:
        """
        Получение текущей ориентации дрона (крен, тангаж, рыскание)
        
        Returns:
            Tuple[float, float, float]: Углы крена, тангажа и рыскания в градусах
        """
        pass
    
    @abstractmethod
    def get_battery_level(self) -> float:
        """
        Получение текущего уровня заряда батареи
        
        Returns:
            float: Уровень заряда батареи в процентах (0-100)
        """
        pass
    
    @abstractmethod
    def start_video_stream(self, resolution: Tuple[int, int] = (1280, 720)) -> bool:
        """
        Запуск видеопотока с камеры дрона
        
        Args:
            resolution: Разрешение видеопотока (ширина, высота)
            
        Returns:
            bool: True если видеопоток успешно запущен, иначе False
        """
        pass
    
    @abstractmethod
    def stop_video_stream(self) -> bool:
        """
        Остановка видеопотока с камеры дрона
        
        Returns:
            bool: True если видеопоток успешно остановлен, иначе False
        """
        pass
    
    @abstractmethod
    def get_video_frame(self) -> Optional[np.ndarray]:
        """
        Получение текущего кадра с камеры дрона
        
        Returns:
            Optional[np.ndarray]: Изображение в формате OpenCV (BGR) или None при ошибке
        """
        pass
    
    @abstractmethod
    def emergency_stop(self) -> bool:
        """
        Аварийная остановка дрона
        
        Returns:
            bool: True если команда успешно отправлена, иначе False
        """
        pass 