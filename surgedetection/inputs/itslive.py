from pathlib import Path
import xarray as xr
import rasterio as rio
import warnings
import pandas as pd

import surgedetection.rasters
import surgedetection.cache

def read_files(crs: rio.crs.CRS, data_path: Path = Path("data/its-live")):

    indices = []
    data = []
    
    variables = ["v", "v_err", "date"]
    variables = {"v": "ice_velocity", "v_err": "ice_velocity_err", "date": "ice_velocity_date"}
    for filepath in data_path.glob("*.nc"):
        
        
        region = filepath.stem.split("_")[0]
        year = int(filepath.stem.split("_")[-1])
        start_date = pd.Timestamp(year=year, month=1, day=1)
        end_date = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)

        for variable in variables:
            variable_path = f"NETCDF:{filepath}:{variable}"
                
            cache_path = surgedetection.cache.get_cache_name("itslive-read_files", [year, variable, data_path]).with_suffix(".vrt")
                
            surgedetection.rasters.create_warped_vrt(variable_path, cache_path, out_crs=crs.to_wkt())
            
                
            indices.append((region, start_date, end_date, variables[variable], "its_live"))
            data.append(cache_path)

        
    return pd.Series(
        data, index=pd.MultiIndex.from_tuples(indices, names=["region", "start_date", "end_date", "kind", "source"])
    ).sort_index()
    
    
    
