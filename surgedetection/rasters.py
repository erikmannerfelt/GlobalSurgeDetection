import copy
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing
import pandas as pd
import rasterio as rio
from affine import Affine
from pyproj import CRS
from tqdm import tqdm
from rasterio.warp import Resampling

import surgedetection.cache


def create_warped_vrt(filepath: Path | str, vrt_filepath: Path | str, out_crs: str, in_crs: str = None) -> None:
    """
    Create a warped VRT from a raster with a different CRS.

    :param filepath: The path to the raster to create a VRT from.
    :param vrt_filepath: The output path of the VRT.
    :param out_crs: The target CRS of the VRT in str format (e.g. WKT)
    """
    import rasterio.warp
    from osgeo import gdal

    ds = gdal.Open(str(filepath))
    vrt = gdal.AutoCreateWarpedVRT(ds, in_crs, out_crs, rasterio.warp.Resampling.bilinear)
    vrt.GetDriver().CreateCopy(str(vrt_filepath), vrt)

    del ds
    del vrt



def build_vrt_new(
     filepaths: Path | str | list[Path | str],
     vrt_filepath: Path,
     dst_bounds: list[float] | None = None,
     dst_res: float | tuple[float, float] | None = None,
     src_crs: CRS | int | str = None,
     dst_crs: CRS | int | str = None,
     resampling: Resampling = Resampling.bilinear,
     multithread: bool = True) -> None:
    from osgeo import gdal

    if isinstance(filepaths, Path):
        filepaths = [str(filepaths)]
    elif isinstance(filepaths, str):
        filepaths = [filepaths]

    crs_s = [src_crs, dst_crs]
    for i, crs in enumerate(crs_s):
        if isinstance(crs, int):
            crs_s[i] = CRS.from_epsg(crs).to_wkt()
        elif isinstance(crs, CRS):
            crs_s[i] = crs.to_wkt()

    if isinstance(dst_res, float) or dst_res is None:
        dst_res = [dst_res] * 2
    
    gdal.Warp(
        str(vrt_filepath),
        list(map(str, filepaths)),
        outputBounds=dst_bounds,
        format="VRT",
        srcSRS=crs_s[0],
        dstSRS=crs_s[1],
        multithread=multithread,
        xRes=dst_res[0],
        yRes=dst_res[1],
        resampleAlg=rio.warp.Resampling.bilinear
    )

def build_vrt(filepaths: list[Path | str], vrt_filepath: Path, gdal_kwargs: dict[str, str] | None = None) -> None:
    from osgeo import gdal
    if gdal_kwargs is None:
        gdal_kwargs = {}

    gdal.BuildVRT(str(vrt_filepath), list(map(str, filepaths)), **gdal_kwargs)


def separate_band_vrt(filepath: Path | str, vrt_filepath: Path | str, band_nrs: list[int], gdal_kwargs: dict[str, str] | None = None) -> None:
    from osgeo import gdal
    if gdal_kwargs is None:
        gdal_kwargs = {}

    ds = gdal.Open(str(filepath))
    gdal.BuildVRT(str(vrt_filepath), ds, bandList=band_nrs, **gdal_kwargs)


def merge_raster_tiles(filepaths: list[str | Path] | list[str] | list[Path], crs: int | CRS, out_path: Path) -> None:

    if out_path.is_file():
        return
    from osgeo import gdal

    temp_dir = tempfile.TemporaryDirectory()

    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)

    filepaths_with_same_crs = []

    for filepath in filepaths:
        with rio.open(filepath) as raster:
            if raster.crs == crs:
                filepaths_with_same_crs.append(filepath)
                continue

        cache_name = surgedetection.cache.get_cache_name("merge_raster_tiles", [filepath, crs, out_path]).with_suffix(
            ".vrt"
        )
        # filename = Path(temp_dir.name).joinpath(filepath.split("/")[-1]).with_suffix(".vrt")

        create_warped_vrt(filepath, cache_name, crs.to_wkt())

        filepaths_with_same_crs.append(cache_name)

    vrt_path = Path(temp_dir.name).joinpath("merged.vrt")

    gdal.BuildVRT(str(vrt_path), list(map(str, filepaths_with_same_crs)))

    os.makedirs(out_path.parent, exist_ok=True)
    if out_path.suffix == ".vrt":
        shutil.copy(vrt_path, out_path)

    elif out_path.suffix == ".tif":
        with tqdm(total=100, desc=f"Mosaicking {out_path.name}") as progress_bar:

            def callback(status: float, _a: Any, _b: Any) -> None:
                progress_bar.update(status * 100 - progress_bar.n)

            gdal.Translate(
                str(out_path),
                str(vrt_path),
                creationOptions=[
                    "COMPRESS=DEFLATE",
                    "TILED=YES",
                    "ZLEVEL=12",
                    "PREDICTOR=3",
                    "NUM_THREADS=ALL_CPUS",
                ],
                callback=callback,
            )

    else:
        raise ValueError(f"Only 'vrt' and 'tif' suffixes are supported. Given: '{out_path.suffix}'")


class RasterParams:
    def __init__(self, transform: Affine, height: int, width: int, crs: CRS, coordinate_suffix: str = ""):

        self.transform = transform
        self.height = int(height)
        self.width = int(width)
        self.crs = CRS(crs)
        self.coordinate_suffix = coordinate_suffix

    def __str__(self) -> str:
        return "\n".join(
            (
                f"RasterParams: {self.bounding_box()} in {self.crs.to_string()}",
                f"Shape: {self.shape()}, xres: {self.xres()}, yres: {self.yres()}",
                f"xarray coordinate suffix: {self.coordinate_suffix}",
            )
        )

    @staticmethod
    def from_bounds(bounding_box: list[float], height: int, width: int, crs: CRS, coordinate_suffix: str = ""):  # type: ignore
        transform = rio.transform.from_bounds(*bounding_box, width, height)
        return RasterParams(transform=transform, height=height, width=width, crs=crs, coordinate_suffix=coordinate_suffix)

    @staticmethod
    def from_bounds_and_res(bounding_box: list[float], xres: float, yres: float, crs: CRS, coordinate_suffix: str = ""):  # type: ignore
        """
        NOTE: The outgoing bounding box may be larger as it is adapted to fit xres/yres evenly
        """
        width = int(np.ceil((bounding_box[2] - bounding_box[0]) / xres))
        height = int(np.ceil((bounding_box[3] - bounding_box[1]) / yres))

        transform = rio.transform.from_origin(bounding_box[0], bounding_box[3], xres, yres)

        return RasterParams(transform=transform, height=height, width=width, crs=crs, coordinate_suffix=coordinate_suffix)

    def copy(self):  # type: ignore
        return copy.deepcopy(self)

    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def bounding_box(self) -> tuple[float, float, float, float]:
        bounds = rio.transform.array_bounds(self.height, self.width, self.transform)
        return (bounds[0], bounds[1], bounds[2], bounds[3])

    def resolution(self) -> tuple[float, float]:
        return (self.transform[0], -self.transform[4])

    def xres(self) -> float:
        return self.resolution()[0]

    def yres(self) -> float:
        return self.resolution()[1]

    def xarray_coords(self) -> list[tuple[str, np.typing.NDArray[np.float64]]]:

        bounds = self.bounding_box()
        res = self.resolution()

        coords = [
            (
                "y" + self.coordinate_suffix,
                np.linspace(bounds[1] + res[1] / 2, bounds[3] - res[1] / 2, self.height, dtype="float64")[::-1],
            ),
            (
                "x" + self.coordinate_suffix,
                np.linspace(
                    bounds[0] + res[0] / 2,
                    bounds[2] - res[0] / 2,
                    self.width,
                    dtype="float64",
                ),
            ),
        ]
        return coords

# class RasterInput:
#     def __init__(
#         self,
#         source: str,
#         start_date: pd.Timestamp,
#         end_date: pd.Timestamp,
#         kind: str,
#         region: str,
#         filepath: Path | str,
#         multi_source: bool = False,
#         multi_date: bool = False,
#         time_prefix: str | None = None,
#         raster_params: RasterParams | None = None,
#         band_n: int = 1,
#     ):
#         self.source = source
#         self.start_date = start_date
#         self.end_date = end_date
#         self.kind = kind
#         self.region = region
#         self.filepath = Path(filepath)
#         self.multi_date = multi_date
#         self.multi_source = multi_source
#         self.time_prefix = time_prefix or kind
#         self.raster_params = raster_params
#         self.band_n = band_n

#     def __str__(self) -> str:
#         return "\n".join(
#             (
#                 f"RasterInput: {self.kind} from {self.source}",
#                 f"Dates: {self.start_date}-{self.end_date}",
#                 f"Region: {self.region}",
#                 f"{self.multi_date=}, {self.multi_source=}",
#             )
#         )


class RasterInput:
    def __init__(
        self,
        sources: str | list[str],
        start_dates: pd.Timestamp | list[pd.Timestamp],
        end_dates: pd.Timestamp | list[pd.Timestamp],
        kinds: str | list[str],
        region: str,
        filepath: Path | str,
        multi_source: bool = False,
        multi_date: bool = False,
        band_numbers: int | list[int] | None = None):

        lists = list(filter(lambda l: isinstance(l[1], list), [("sources", sources), ("start_dates", start_dates), ("end_dates", end_dates), ("kinds", kinds)]))
        keys_with_lists = [l[0] for l in lists]
        if len(lists) > 0:
            list_lengths = [len(o) for k, o in lists]

            if any(all(s in keys_with_lists and s2 in keys_with_lists for s in ["sources", "kinds"]) for s2 in ["start_date", "end_date"]):
                raise ValueError("Multi-source/kind and multi-date inputs are currently not supported")

            if band_numbers is None:
                band_numbers = list(range(1, max(list_lengths) + 1)) 

            for key, value in lists:
                if key in ["start_dates", "end_dates"]:
                    multi_date = True

                elif key in ["sources"]:
                    multi_source = True
            
            first = list_lengths[0]
            for i, length in enumerate(list_lengths[1:], start=1):
                if first[1] != length[1]:
                    raise ValueError(f"Length of {first[0]} is inconsistent with {length[0]}")
        else:
            band_numbers = [1]

        if "kinds" not in keys_with_lists:
            kinds = [kinds]
        if "sources" not in keys_with_lists:
            sources = [sources]
        if "end_dates" not in keys_with_lists:
            end_dates = [end_dates]
        if "start_dates" not in keys_with_lists:
            start_dates = [start_dates]
            

        self.sources: list[str] = sources
        self.start_dates: list[pd.Timestamp] = start_dates
        self.end_dates: list[pd.Timestamp] = end_dates
        self.kinds: list[str] = kinds
        self.region = region
        self.filepath = Path(filepath)
        self.multi_source = multi_source
        self.multi_date = multi_date
        self.band_numbers: list[int] = band_numbers


    def __str__(self) -> str:
        return "\n".join(
            (
                f"RasterInput: {self.kinds} from {self.sources}",
                f"Dates: {self.start_dates}-{self.end_dates}",
                f"Region: {self.region}",
                f"{self.multi_date=}, {self.multi_source=}, {self.band_numbers=}",
            )
        )



def make_input_series(inputs: list[RasterInput]) -> pd.Series:

    indices = []
    data = []
    names = ["region", "start_date", "end_date", "kind", "source", "multi_date", "multi_source"]
    for r_input in inputs:
        data.append(r_input.filepath)
        indices.append([getattr(r_input, name) for name in names])

    series = pd.Series(data, index=pd.MultiIndex.from_tuples(indices, names=names))

    return series
