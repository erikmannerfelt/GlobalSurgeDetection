import re
from pathlib import Path

import pandas as pd
import rasterio as rio
from pyproj import CRS

import surgedetection.cache
import surgedetection.io
import surgedetection.rasters
from surgedetection.constants import CONSTANTS


def get_filepaths(tarfile_dir: str = "hugonnet-etal-2021/", crs: int | CRS = 32633) -> pd.Series:

    full_tarfile_dirpath = CONSTANTS.data_path.joinpath(tarfile_dir)

    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)

    indices = []
    data = []
    for filepath in full_tarfile_dirpath.glob("*.tar"):
        region = filepath.stem.split("_")[0]
        start_date = pd.to_datetime(filepath.stem.split("_")[-2])
        end_date = pd.to_datetime(filepath.stem.split("_")[-1])

        indices += [(region, start_date, end_date, kind, "hugonnet-etal-2021") for kind in ["dhdt", "dhdt_err"]]

        data.append(load_tarfile(filepath, crs, pattern=r".*dhdt\.tif"))
        data.append(load_tarfile(filepath, crs, pattern=r".*dhdt_err\.tif"))

    return pd.Series(
        data,
        index=pd.MultiIndex.from_tuples(indices, names=["region", "start_date", "end_date", "kind", "source"]),
        dtype=object,
    ).sort_index()


def load_tarfile(
    filepath: Path,
    crs: CRS,
    pattern: str = r".*\.tif",
) -> Path:
    cache_filename = surgedetection.cache.get_cache_name("load_tarfile", args=[filepath, pattern, crs]).with_suffix(
        ".vrt"
    )

    if cache_filename.is_file():
        return cache_filename

    files = surgedetection.io.list_tar_filepaths(filepath, pattern=pattern, prepend_vsitar=True)

    surgedetection.rasters.merge_raster_tiles(
        filepaths=files,
        crs=crs,
        out_path=cache_filename,
    )

    return cache_filename
