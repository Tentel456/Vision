from .base_connector import DroneConnector
from .dji_connector import DJIConnector
from .mavlink_connector import MAVLinkConnector
from .parrot_connector import ParrotConnector

# Словарь доступных коннекторов для разных типов дронов
AVAILABLE_CONNECTORS = {
    'dji': DJIConnector,
    'mavlink': MAVLinkConnector,
    'ardupilot': MAVLinkConnector,
    'px4': MAVLinkConnector,
    'parrot': ParrotConnector
}

def get_connector(drone_type, **kwargs):
    """
    Фабричный метод для получения соответствующего коннектора дрона
    
    Args:
        drone_type (str): Тип дрона ('dji', 'mavlink', 'ardupilot', 'px4', 'parrot')
        **kwargs: Дополнительные параметры для инициализации коннектора
        
    Returns:
        DroneConnector: Экземпляр соответствующего коннектора
    """
    if drone_type not in AVAILABLE_CONNECTORS:
        raise ValueError(f"Неподдерживаемый тип дрона: {drone_type}. "
                         f"Доступные типы: {list(AVAILABLE_CONNECTORS.keys())}")
    
    return AVAILABLE_CONNECTORS[drone_type](**kwargs) 