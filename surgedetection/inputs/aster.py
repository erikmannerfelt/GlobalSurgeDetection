from pathlib import Path

import pandas as pd
from pyproj import CRS

import surgedetection.cache
import surgedetection.io
import surgedetection.rasters
from surgedetection.rasters import RasterInput
from surgedetection.constants import CONSTANTS


def get_filepaths(tarfile_dir: str = "hugonnet-etal-2021/", crs: int | CRS = 32633) -> list[RasterInput]:

    full_tarfile_dirpath = CONSTANTS.data_path.joinpath(tarfile_dir)

    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)

    rasters = []
    for filepath in full_tarfile_dirpath.glob("*.tar"):
        region = filepath.stem.split("_")[0]
        start_date = pd.to_datetime(filepath.stem.split("_")[-2])
        end_date = pd.to_datetime(filepath.stem.split("_")[-1])

        for kind in ["dhdt", "dhdt_err"]:
            rasters.append(
                RasterInput(
                    source="hugonnet-etal-2021",
                    start_date=start_date,
                    end_date=end_date,
                    kind=kind,
                    region=region,
                    filepath=load_tarfile(filepath, crs, pattern=".*" + kind + r"\.tif"),
                    multi_date=True,
                    multi_source=False,
                    time_prefix="dhdt",
                )

            )
        #indices += [(region, start_date, end_date, kind, "hugonnet-etal-2021") for kind in ["dhdt", "dhdt_err"]]

        #data.append()
        #data.append(load_tarfile(filepath, crs, pattern=r".*dhdt_err\.tif"))

    return rasters



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
