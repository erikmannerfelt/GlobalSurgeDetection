from pyproj import CRS

import surgedetection.inputs.aster
import surgedetection.inputs.itslive
import surgedetection.inputs.sar
import surgedetection.inputs.dem
from surgedetection.rasters import RasterInput


def get_all_rasters(crs: CRS) -> list[RasterInput]:

    series = (
        surgedetection.inputs.dem.load_dem(crs=crs)
        + surgedetection.inputs.aster.get_filepaths(crs=crs)
        + surgedetection.inputs.itslive.read_files(crs=crs)
        + surgedetection.inputs.sar.read_all_sar(crs=crs)
    )

    return series
