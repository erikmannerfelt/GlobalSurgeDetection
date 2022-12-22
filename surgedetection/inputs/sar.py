import re
import zipfile
from pathlib import Path

import pandas as pd
import rasterio as rio
from pyproj import CRS

import surgedetection.cache
import surgedetection.rasters

from surgedetection.rasters import RasterInput
from surgedetection.constants import CONSTANTS


def read_all_sar(crs: CRS | int, data_path: str = "sar") -> list[RasterInput]:
    return (
        read_sentinel1_diff(crs=crs, data_path=data_path + "/sentinel-1-diff") + 
        read_asar_jers(crs=crs, data_path=data_path)
        #read_sentinel1(crs=crs, data_path=data_path + "/sentinel-1") + 
    )


def read_sentinel1_diff(crs: CRS | int, data_path: str = "sar/sentinel-1-diff") -> list[RasterInput]:
    full_data_path = CONSTANTS.data_path.joinpath(data_path)

    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)

    rasters = []
    for filepath in full_data_path.glob("*.tif"):
        region_id = filepath.stem.split("-")[0]

        year_before, year_after = map(int, filepath.stem.split("_")[-1].split("-"))
        start_date = pd.Timestamp(year=year_before - 1, month=10, day=1)
        end_date = pd.Timestamp(year=year_after, month=9, day=30)

        with rio.open(filepath) as raster:
            band_names = raster.descriptions
        for i, band_name in enumerate(band_names, start=1):
            if band_name == "mean":
                continue

            vrt_filepath = surgedetection.cache.get_cache_name(
                "sar-read_sentinel1_diff", args=[year_before, year_after, filepath, band_name, crs], extension=".vrt",
            )
            if not vrt_filepath.is_file():
                surgedetection.rasters.separate_band_vrt(filepath, vrt_filepath, band_nrs=[i])

            rasters.append(RasterInput(
                source="S1-" + band_name,
                start_date=start_date,
                end_date=end_date,
                kind="sar_backscatter_diff",
                region=region_id,
                filepath=vrt_filepath,
                multi_source=True,
                multi_date=True,
            ))
    return rasters

def read_sentinel1(crs: CRS | int, data_path: str = "sar/sentinel-1") -> list[RasterInput]:
    full_data_path = CONSTANTS.data_path.joinpath(data_path)

    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)
    #indices = []
    #data = []

    rasters = []
    for filepath in full_data_path.glob("*.tif"):
        stem_split = filepath.stem.split("-")

        region = stem_split[0]
        satellite = "-".join(stem_split[1:4]).upper()

        year = int(stem_split[-1])

        start_date = pd.Timestamp(year=year - 1, month=10, day=1)
        end_date = pd.Timestamp(year=year, month=4, day=1)

        with rio.open(filepath) as raster:
            if raster.crs == crs:
                out_path = filepath

            else:
                cache_path = surgedetection.cache.get_cache_name(
                    "sar-read_sentinel1", [year, satellite, data_path]
                ).with_suffix(".vrt")
                surgedetection.rasters.create_warped_vrt(filepath, cache_path, out_crs=crs.to_wkt())

                out_path = cache_path

        
        rasters.append(
            RasterInput(
                source=satellite,
                start_date=start_date,
                end_date=end_date,
                kind="sar_backscatter",
                region=region,
                filepath=out_path,
                multi_source=True,
                multi_date=True,
            )
        )

        #indices.append((region, start_date, end_date, "sar_backscatter", satellite))

    #return pd.Series(
    #    data,
    #    index=pd.MultiIndex.from_tuples(indices, names=["region", "start_date", "end_date", "kind", "source"]),
    #    dtype=object,
    #).sort_index()
    return rasters


def read_asar_jers(crs: CRS | int, data_path: str = "sar") -> list[RasterInput]:
    full_data_path = CONSTANTS.data_path.joinpath(data_path)

    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)

    #indices = []
    #data: list[str | Path] = []
    rasters = []
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

            #indices.append(("svalbard", start_date, end_date, "sar_backscatter", satellite))
            with rio.open(full_fp) as raster:
                if raster.crs == crs:
                    out_path = full_fp
                    #data.append(full_fp)

                else:
                    cache_path = surgedetection.cache.get_cache_name(
                        "sar-read_asar_jers", [start_date, end_date, data_path]
                    ).with_suffix(".vrt")
                    surgedetection.rasters.create_warped_vrt(full_fp, cache_path, out_crs=crs.to_wkt())

                    out_path = full_fp
                    #data.append(cache_path)

            rasters.append(RasterInput(
                region="svalbard",
                source=satellite,
                start_date=start_date,
                end_date=end_date,
                kind="sar_backscatter",
                filepath=out_path,
                multi_source=True,
                multi_date=True
            ))
    return rasters
    #return pd.Series(
    #    data, index=pd.MultiIndex.from_tuples(indices, names=["region", "start_date", "end_date", "kind", "source"])
    #).sort_index()
