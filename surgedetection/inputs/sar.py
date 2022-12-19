import re
import zipfile
from pathlib import Path

import pandas as pd
import rasterio as rio
from pyproj import CRS

import surgedetection.cache
import surgedetection.rasters
from surgedetection.constants import CONSTANTS


def read_all_sar(crs: CRS | int, data_path: str = "/sar") -> pd.DataFrame:
    return pd.concat(
        [
            read_sentinel1(crs=crs, data_path=data_path + "/sentinel-1"),
            read_asar_jers(crs=crs, data_path=data_path),
        ]
    )


def read_sentinel1(crs: CRS | int, data_path: str = "/sar/sentinel-1") -> pd.Series:
    full_data_path = CONSTANTS.data_path.joinpath(data_path)

    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)
    indices = []
    data = []

    for filepath in full_data_path.glob("*.tif"):
        stem_split = filepath.stem.split("-")

        region = stem_split[0]
        satellite = "-".join(stem_split[1:4]).upper()

        year = int(stem_split[-1])

        start_date = pd.Timestamp(year=year - 1, month=10, day=1)
        end_date = pd.Timestamp(year=year, month=4, day=1)

        with rio.open(filepath) as raster:
            if raster.crs == crs:
                data.append(filepath)

            else:
                cache_path = surgedetection.cache.get_cache_name(
                    "sar-read_sentinel1", [year, satellite, data_path]
                ).with_suffix(".vrt")
                surgedetection.rasters.create_warped_vrt(filepath, cache_path, out_crs=crs.to_wkt())

                data.append(cache_path)

        indices.append((region, start_date, end_date, "sar_backscatter", satellite))

    return pd.Series(
        data, index=pd.MultiIndex.from_tuples(indices, names=["region", "start_date", "end_date", "kind", "source"])
    ).sort_index()


def read_asar_jers(crs: CRS | int, data_path: str = "/sar") -> pd.Series:
    full_data_path = CONSTANTS.data_path.joinpath(data_path)
    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)

    indices = []
    data: list[str | Path] = []
    for zip_filepath in [full_data_path.joinpath("Svalbard_ASAR.zip"), full_data_path.joinpath("Svalbard_JERS.zip")]:

        for filename in zipfile.ZipFile(zip_filepath).namelist():
            if not any(
                re.match(pattern, filename) is not None
                for pattern in [r".*asar.tflt.ortho.tif$", r".*JERS-1_1.*", r".*ERS-._1.*"]
            ):
                continue

            full_fp = f"/vsizip/{zip_filepath}/{filename}"

            if "asar" in filename:
                end_date = pd.to_datetime(filename.split("/")[-1][:6], format="%Y%m")
                start_date = end_date - pd.Timedelta(days=150)

                satellite = "ASAR"

            elif "JERS-1" in filename:
                year_range = filename.split("/")[-1].split("_")[-1].replace(".tif", "").split("-")
                start_date = pd.Timestamp(year=int(year_range[0]), month=1, day=1)
                end_date = pd.Timestamp(year=int(year_range[1]), month=12, day=31)

                satellite = "JERS-1"
            elif "ERS-" in filename:
                end_date = pd.Timestamp(year=int(filename.split("_")[-1].replace(".tif", "")), month=4, day=1)
                start_date = end_date - pd.Timedelta(days=150)

                satellite = filename.split("/")[-1][:5]
            else:
                raise NotImplementedError()

            indices.append(("svalbard", start_date, end_date, "sar_backscatter", satellite))
            with rio.open(full_fp) as raster:
                if raster.crs == crs:
                    data.append(full_fp)

                else:
                    cache_path = surgedetection.cache.get_cache_name(
                        "sar-read_asar_jers", [start_date, end_date, data_path]
                    ).with_suffix(".vrt")
                    surgedetection.rasters.create_warped_vrt(full_fp, cache_path, out_crs=crs.to_wkt())

                    data.append(cache_path)
    return pd.Series(
        data, index=pd.MultiIndex.from_tuples(indices, names=["region", "start_date", "end_date", "kind", "source"])
    ).sort_index()
