from pathlib import Path

import surgedetection.utilities


class Constants(surgedetection.utilities.ConstantType):
    """Readonly constants"""

    pixel_size: float = 100.0  # The horizontal and vertical resolution
    data_path: Path = Path(__file__).joinpath("../data")
    manual_input_data_path: Path = Path(__file__).joinpath("../manual_input")


CONSTANTS = Constants()
