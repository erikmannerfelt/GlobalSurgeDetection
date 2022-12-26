"""Functions to import and format the Randolph Glacier Inventory (RGI)."""
import geopandas as gpd
import pandas as pd
import urllib
import zipfile

import surgedetection.cache
from surgedetection.constants import CONSTANTS

BASE_DOWNLOAD_URL = "http://www.glims.org/RGI/rgi60_files/00_rgi60.zip"
DATA_DIR = CONSTANTS.data_path.joinpath("rgi")
RGI6_SUBDIR = DATA_DIR.joinpath("rgi6")

def read_rgi6(regions: int | list[int], query: str | None = None) -> gpd.GeoDataFrame:
    """
    Read RGI6 outlines for the given region(s).

    Arguments
    ---------
    regions: One or multiple RGI regions to read outlines from.
    query: Optional. A query to filter the data.

    Returns
    -------
    A concatenated GeoDataFrame of all specified RGI regions.
    """
    # Convert the region argument to a list if only one region is given
    if not isinstance(regions, list):
        regions = [regions]

    region_data = []
    for region in regions:
        for _ in range(2):
            matching = list(RGI6_SUBDIR.glob(f"{str(region).zfill(2)}_*"))
            if len(matching) == 0:
                download_rgi6()

        if len(matching) == 0:
            raise ValueError(f"No file found for RGI region {region} in {data_path}")

        filepath = matching[0]
        inner_filename = filepath.stem.replace("nsidc0770_", "") + ".shp"
        data = gpd.read_file(f"/vsizip/{filepath}/{inner_filename}")
        if query is not None:
            data.query(query, inplace=True)
        region_data.append(data)

    return pd.concat(region_data, ignore_index=True)


def read_all_rgi6(query: str | None = None) -> gpd.GeoDataFrame:
    """
    Read all RGI6 outlines.

    Arguments
    ---------
    data_path: The path (relative to the data directory) of the RGI outlines.
    query: Optional. A query to filter the data.

    Returns
    -------
    A GeoDataFrame of all RGI6 outlines.
    """
    cache_path = surgedetection.cache.get_cache_name("read_all_rgi6", [RGI6_SUBDIR, query]).with_suffix(".feather")
    if cache_path.is_file():
        return gpd.read_feather(cache_path)
    rgi = pd.concat((read_rgi6(region, query=query) for region in CONSTANTS.rgi_regions), ignore_index=True)

    rgi.to_feather(cache_path)
    return rgi


def read_rgi6_regions() -> gpd.GeoDataFrame:
    """
    Read the predefined RGI6 (O2) regions.

    Returns
    -------
    A GeoDataFrame of all RGI6 region outlines.
    """
    filepath = RGI6_SUBDIR.joinpath("00_rgi60_regions.zip")
    if not filepath.is_file():
        download_rgi6()

    filename = filepath.stem.replace("nsidc0770_", "").replace("regions", "O2Regions") + ".shp"
    return gpd.read_file(f"/vsizip/{filepath}/{filename}")


def download_rgi6():
    filename ="00_rgi60.zip" 
    filepath = DATA_DIR.joinpath(filename)

    if filepath.is_file():
        with zipfile.ZipFile(filepath) as zipped:
            filepaths = [RGI6_SUBDIR.joinpath(p) for p in zipped.namelist()] 

        if all(fp.is_file() for fp in filepaths):
            return

    if not filepath.is_file():
        print("Downloading RGI6 outlines")
        surgedetection.utilities.download_file(urllib.parse.urljoin(BASE_DOWNLOAD_URL, filename), filepath)

    with zipfile.ZipFile(filepath) as zipped:
        zipped.extractall(RGI6_SUBDIR)


    
