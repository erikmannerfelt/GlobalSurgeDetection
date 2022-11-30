import rasterio as rio
from pathlib import Path
import surgedetection.io
import surgedetection.cache
import surgedetection.rasters
import re
import pandas as pd

def get_filepaths(tarfile_dir: Path = Path("data/hugonnet-etal-2021/"), crs: int | rio.crs.CRS = 32633) -> pd.Series:
    filepaths = [Path("data/hugonnet-etal-2021/07_rgi60_2000-01-01_2005-01-01.tar")]
    if isinstance(crs, int):
        crs = rio.crs.CRS.from_epsg(crs)
    
    indices = []
    data = []
    for filepath in tarfile_dir.glob("*.tar"):
        region = filepath.stem.split("_")[0]
        start_date = pd.to_datetime(filepath.stem.split("_")[-2])
        end_date = pd.to_datetime(filepath.stem.split("_")[-1])
        

        indices += [(region, start_date, end_date, kind, "hugonnet-etal-2021") for kind in ["dhdt", "dhdt_err"]]

    
        data.append(load_tarfile(filepath, crs, pattern=r".*dhdt\.tif"))
        data.append(load_tarfile(filepath, crs, pattern=r".*dhdt_err\.tif"))
        
    return pd.Series(data, index=pd.MultiIndex.from_tuples(indices, names=["region", "start_date", "end_date", "kind", "source"])).sort_index()

        
def load_tarfile(filepath: Path, crs: rio.crs.CRS, pattern: str = r".*\.tif",) -> Path:
    cache_filename = surgedetection.cache.get_cache_name("load_tarfile", args=[filepath, pattern, crs]).with_suffix(".vrt")
    
    if cache_filename.is_file():
        return cache_filename
        
    
    files = surgedetection.io.list_tar_filepaths(filepath, pattern=pattern, prepend_vsitar=True)
    
    surgedetection.rasters.merge_raster_tiles(
        filepaths=files,
        crs=crs,
        out_path=cache_filename,
    )
    
    
    return cache_filename

