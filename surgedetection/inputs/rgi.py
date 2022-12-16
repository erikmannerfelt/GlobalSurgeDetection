from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import surgedetection.cache

def read_rgi6(regions: int | list[int], data_path: Path = Path("data/rgi/rgi6")):
    if not isinstance(regions, list):
        regions = [regions]

    region_data = []
    for region in regions:
        matching = list(data_path.glob(f"nsidc0770_{str(region).zfill(2)}_*"))
        if len(matching) == 0:
            raise ValueError(f"No file found for RGI region {region} in {data_path}")

        filepath = matching[0]

        inner_filename = filepath.stem.replace("nsidc0770_", "") + ".shp"
        

        region_data.append(gpd.read_file(f"/vsizip/{filepath}/{inner_filename}"))
        

    return pd.concat(region_data)

def read_all_rgi6(data_path: Path = Path("data/rgi/rgi6")) -> gpd.GeoDataFrame:
    cache_path = surgedetection.cache.get_cache_name(
        "rgi6", [data_path]
    ).with_suffix(".feather")
    if cache_path.is_file():
        return gpd.read_feather(cache_path)
    rgi = pd.concat((read_rgi6(region, data_path=data_path) for region in range(1, 20)))

    rgi.to_feather(cache_path)
    return rgi
    
    
def read_rgi6_regions(filepath: Path = Path("data/rgi/rgi6/nsidc0770_00_rgi60_regions.zip")) -> gpd.GeoDataFrame:
    filename = "00_rgi60_O2Regions.shp"
    return gpd.read_file(f"/vsizip/{filepath}/{filename}")
