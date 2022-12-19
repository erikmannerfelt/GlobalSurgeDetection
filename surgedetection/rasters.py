import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

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


def merge_raster_tiles(filepaths: list[str | Path] | list[str] | list[Path], crs: int | CRS, out_path: Path) -> None:

    if out_path.is_file():
        return
    import rasterio as rio
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
