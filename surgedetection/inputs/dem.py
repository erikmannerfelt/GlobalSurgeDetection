from pathlib import Path
from pyproj import CRS
import pandas as pd

import surgedetection.rasters
import surgedetection.cache
from surgedetection.rasters import RasterInput

def load_dem(crs: int | CRS, url: str = "/vsicurl/https://schyttholmlund.com/share/COP90_hh_merged_higher_z.tif") -> list[RasterInput]:

    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)

    cache_path = surgedetection.cache.get_cache_name("load_dem", [crs, url], extension="vrt")

    if crs.to_epsg() != 32633:
        raise NotImplementedError("The DEM is currently hardcoded to Svalbard")
    #surgedetection.rasters.create_warped_vrt(url, cache_path, crs.to_wkt())
    cache_path = Path("data/dem/COP90-Svalbard.tif")

    return [surgedetection.rasters.RasterInput(
        source="copernicus-glo90",
        start_date=pd.Timestamp("2011-01-01"),
        end_date=pd.Timestamp("2015-07-01"),
        kind="dem",
        region="07",
        filepath=cache_path,
        multi_source=False,
        multi_date=False,
    )]
