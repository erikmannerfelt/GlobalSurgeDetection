import pandas as pd

import surgedetection.inputs.aster
import surgedetection.inputs.dem
import surgedetection.inputs.itslive
import surgedetection.inputs.sar
from surgedetection.rasters import RasterInput


def get_all_rasters(region: pd.Series) -> list[RasterInput]:

    series = (
        [surgedetection.inputs.dem.load_region_dem(region=region)]
        + surgedetection.inputs.aster.get_files(region=region)
        + surgedetection.inputs.itslive.get_files(region=region)
        + surgedetection.inputs.sar.get_sentinel1_diff_files(region=region)
    )

    return series
