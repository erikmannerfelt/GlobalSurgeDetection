import urllib
import warnings
from pathlib import Path

import pandas as pd
import rasterio as rio
import xarray as xr
from pyproj import CRS
from tqdm import tqdm

import surgedetection.cache
import surgedetection.rasters
import surgedetection.utilities
from surgedetection.constants import CONSTANTS
from surgedetection.rasters import RasterInput

DATA_DIR = CONSTANTS.data_path.joinpath("its-live")
BASE_DOWNLOAD_URL = "https://its-live-data.s3.amazonaws.com/velocity_mosaic/landsat/v00.0/annual/"
YEARS = list(range(1985, 2019))


def get_files(region: pd.Series) -> list[RasterInput]:

    rgi_regions = list(map(lambda s: int(s.split("-")[0]), region["rgi_regions"].split("/")))
    filepaths = download_all_itslive(rgi_regions=rgi_regions)

    rasters = []
    variables = {"v": "ice_velocity", "v_err": "ice_velocity_err", "date": "ice_velocity_date"}
    for filepath in filepaths:

        itslive_region = filepath.stem.split("_")[0]
        year = int(filepath.stem.split("_")[-1])
        start_date = pd.Timestamp(year=year, month=1, day=1)
        end_date = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)

        for variable in variables:
            variable_path = f"NETCDF:{filepath}:{variable}"

            cache_path = surgedetection.cache.get_cache_name(
                "itslive-read_files", [itslive_region, year, variable, DATA_DIR]
            ).with_suffix(".vrt")

            if not cache_path.is_file():
                surgedetection.rasters.create_warped_vrt(variable_path, cache_path, out_crs=region["crs"].to_wkt())

            rasters.append(
                RasterInput(
                    source="its-live",
                    start_date=start_date,
                    end_date=end_date,
                    kind=variables[variable],
                    region=itslive_region,
                    filepath=cache_path,
                    multi_source=False,
                    multi_date=True,
                    time_prefix=variables["v"],
                )
            )

    return rasters


def rgi_mapping(rgi_region: int | str, res: str = "_G0240") -> str | None:
    if isinstance(rgi_region, str):
        rgi_region = int(rgi_region)

    match rgi_region:
        case 1 | 2:
            return "ALA" + res
        case 3:
            return "CAN" + res
        case 5:
            return "GRE" + res
        case 6:
            return "ICE" + res
        case 7 | 9:
            return "SRA" + res
        case 13 | 14 | 15:
            return "HMA" + res
        case 17:
            return "PAT" + res
        case 19:
            return "ANT" + res

    return None


def download_all_itslive(rgi_regions: list[int] | None = None, progress: bool = True) -> list[Path]:
    labels = set(filter(None, map(rgi_mapping, rgi_regions or CONSTANTS.rgi_regions)))

    filepaths = []
    queries = []
    for label in labels:
        for year in YEARS:
            filename = f"{label}_{year}.nc"
            filepath = DATA_DIR.joinpath(filename)
            if filepath.is_file():
                filepaths.append(filepath)
            else:
                queries.append({"url": urllib.parse.urljoin(BASE_DOWNLOAD_URL, filename), "filepath": filepath})

    for query in tqdm(queries, desc="Downloading ITS-LIVE data", disable=(not progress or len(queries) == 0)):
        filepath = surgedetection.utilities.download_file(**query, progress=False)

        # Validate that the filepath is a valid nc file
        xr.open_dataset(filepath).close()
        filepaths.append(filepath)

    return filepaths
