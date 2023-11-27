"""Constant values that are used throughout the package."""
from pathlib import Path

import surgedetection.utilities


class Constants(surgedetection.utilities.ConstantType):
    """Readonly constants that cannot be changed accidentally."""

    pixel_size: float = 100.0  # The horizontal and vertical resolution of "high-res" datasets
    lowres_pixel_size: float = 25000.0  # The horizontal and vertical resolution of low-res datasets (e.g. ERA5)
    data_path: Path = Path(__file__).parent.parent.joinpath("data")  # The path to the directory with input data
    manual_input_data_path: Path = Path(__file__).parent.parent.joinpath("manual_input")  # The path to the directory with steps that require manual input
    output_dir_path: Path = Path(__file__).parent.parent.joinpath("output")  # The path to the output data directory
    days_per_year: float = 365.2425  # According to Wikipedia
    rgi_regions: list[int] = list(range(1, 20))  # The RGI6 region numbers 
    sentinel1_years: list[int] = list(range(2015, 2023)) # Years where Sentinel-1 data are available
    era5_years: list[int] = list(range(1985, 2023))  # Years where ERA5 data should be downloaded


CONSTANTS = Constants()
