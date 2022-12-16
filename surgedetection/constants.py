
import surgedetection.utilities


class Constants(surgedetection.utilities.ConstantType):
    """Readonly constants"""
    pixel_size: float = 100.  # The horizontal and vertical resolution
    ...


CONSTANTS = Constants()