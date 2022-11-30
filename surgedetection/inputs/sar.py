import rasterio as rio
from pathlib import Path
import pandas as pd
import zipfile

import surgedetection.rasters
import surgedetection.cache

def read_sentinel1(crs: rio.crs.CRS | int, data_path=Path("data/sar/sentinel-1")):


    if isinstance(crs, int):
        crs = rio.crs.CRS.from_epsg(crs)
    indices = []
    data = []
    
    for filepath in data_path.glob("*.tif"):
        stem_split = filepath.stem.split("-")
        
        region = stem_split[0]
        satellite = "-".join(stem_split[1:4]).upper()
        
        year = int(stem_split[-1])
        
        start_date = pd.Timestamp(year=year - 1, month=10, day=1)
        end_date = pd.Timestamp(year=year, month=4, day=1)
        
        with rio.open(filepath) as raster:
            if raster.crs == crs:
                data.append(filepath)
            
            else:
                cache_path = surgedetection.cache.get_cache_name("sar-read_sentinel1", [year, satellite, data_path]).with_suffix(".vrt")
                surgedetection.rasters.create_warped_vrt(filepath, cache_path, out_crs=crs.to_wkt())
                
                data.append(cache_path)
        
        indices.append((region, start_date, end_date, "sar_backscatter", satellite))

                

    return pd.Series(
        data, index=pd.MultiIndex.from_tuples(indices, names=["region", "start_date", "end_date", "kind", "source"])
    ).sort_index()
    

def read_asar_jers(crs: rio.crs.CRS | int, data_path=Path("data/sar")):

    asar_filepath = data_path.joinpath("Svalbard_ASAR.zip")
    jers_filepath = data_path.joinpath("Svalbard_JERS.zip")
    
    asar_files = zipfile.ZipFile(asar_filepath).namelist()
    
    jers_files = zipfile.ZipFile(jers_filepath).namelist()
    
    
    
    
    
    
    
