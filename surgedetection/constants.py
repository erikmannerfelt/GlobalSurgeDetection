"""Constant values that are used throughout the package."""
from pathlib import Path

import surgedetection.utilities


class Constants(surgedetection.utilities.ConstantType):
    """Readonly constants that cannot be changed accidentally."""

    pixel_size: float = 100.0  # The horizontal and vertical resolution of "high-res" datasets
    lowres_pixel_size: float = 25000.0  # The horizontal and vertical resolution of low-res datasets (e.g. ERA5)
    data_path: Path = Path(__file__).parent.parent.joinpath("data")
    manual_input_data_path: Path = Path(__file__).parent.parent.joinpath("manual_input")
    output_dir_path: Path = Path(__file__).parent.parent.joinpath("output")
    days_per_year: float = 365.2425
    rgi_regions: list[int] = list(range(1, 20))
    sentinel1_years: list[int] = list(range(2015, 2023))
    era5_years: list[int] = list(range(1985, 2023))


CONSTANTS = Constants()
