import pandas as pd
import rasterio as rio

import surgedetection.inputs.aster
import surgedetection.inputs.itslive
import surgedetection.inputs.sar

def get_all_paths(crs: rio.crs.CRS) -> pd.Series:

    series = pd.concat([
        surgedetection.inputs.aster.get_filepaths(crs=crs),
        surgedetection.inputs.itslive.read_files(crs=crs),
        surgedetection.inputs.sar.read_sentinel1(crs=crs),
    ])
    
    return series
    
