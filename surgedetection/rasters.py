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

import surgedetection.cache


def create_warped_vrt(filepath: Path | str, vrt_filepath: Path | str, out_crs: str) -> None:
    """
    Create a warped VRT from a raster with a different CRS.

    :param filepath: The path to the raster to create a VRT from.
    :param vrt_filepath: The output path of the VRT.
    :param out_crs: The target CRS of the VRT in str format (e.g. WKT)
    """
    import rasterio.warp
    from osgeo import gdal

    ds = gdal.Open(str(filepath))
    vrt = gdal.AutoCreateWarpedVRT(ds, None, out_crs, rasterio.warp.Resampling.cubic_spline)
    vrt.GetDriver().CreateCopy(str(vrt_filepath), vrt)

    del ds
    del vrt


def separate_band_vrt(filepath: Path | str, vrt_filepath: Path | str, band_nrs: list[int]) -> None:
    from osgeo import gdal

    ds = gdal.Open(str(filepath))
    gdal.BuildVRT(str(vrt_filepath), ds, bandList=band_nrs)


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


class RasterInput:
    def __init__(
        self,
        source: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        kind: str,
        region: str,
        filepath: Path | str,
        multi_source: bool = False,
        multi_date: bool = False,
        time_prefix: str | None = None,
    ):
        self.source = source
        self.start_date = start_date
        self.end_date = end_date
        self.kind = kind
        self.region = region
        self.filepath = Path(filepath)
        self.multi_date = multi_date
        self.multi_source = multi_source
        self.time_prefix = time_prefix or kind

    def __str__(self) -> str:
        return "\n".join(
            (
                f"RasterInput: {self.kind} from {self.source}",
                f"Dates: {self.start_date}-{self.end_date}",
                f"Region: {self.region}",
                f"{self.multi_date=}, {self.multi_source=}",
            )
        )


class RasterParams:
    def __init__(self, transform: Affine, height: int, width: int, crs: CRS):

        self.transform = transform
        self.height = height
        self.width = width
        self.crs = crs

    @staticmethod
    def from_bounds(bounding_box: list[float], height: int, width: int, crs: CRS):  # type: ignore
        transform = rio.transform.from_bounds(*bounding_box, width, height)
        return RasterParams(transform=transform, height=height, width=width, crs=crs)

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
                "y",
                np.linspace(bounds[1] + res[1] / 2, bounds[3] - res[1] / 2, self.height, dtype="float64")[::-1],
            ),
            (
                "x",
                np.linspace(
                    bounds[0] + res[0] / 2,
                    bounds[2] - res[0] / 2,
                    self.width,
                    dtype="float64",
                ),
            ),
        ]
        return coords


def make_input_series(inputs: list[RasterInput]) -> pd.Series:

    indices = []
    data = []
    names = ["region", "start_date", "end_date", "kind", "source", "multi_date", "multi_source"]
    for r_input in inputs:
        data.append(r_input.filepath)
        indices.append([getattr(r_input, name) for name in names])

    series = pd.Series(data, index=pd.MultiIndex.from_tuples(indices, names=names))

    return series
