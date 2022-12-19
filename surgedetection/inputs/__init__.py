import pandas as pd
from pyproj import CRS

import surgedetection.inputs.aster
import surgedetection.inputs.itslive
import surgedetection.inputs.sar


def get_all_paths(crs: CRS) -> pd.Series:

    series = pd.concat(
        [
            surgedetection.inputs.aster.get_filepaths(crs=crs),
            surgedetection.inputs.itslive.read_files(crs=crs),
            surgedetection.inputs.sar.read_all_sar(crs=crs),
        ]
    )

    return series
