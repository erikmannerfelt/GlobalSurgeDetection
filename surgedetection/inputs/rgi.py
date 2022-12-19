"""Functions to import and format the Randolph Glacier Inventory (RGI)."""
import geopandas as gpd
import pandas as pd

import surgedetection.cache
from surgedetection.constants import CONSTANTS


def read_rgi6(regions: int | list[int], data_path: str = "rgi/rgi6") -> gpd.GeoDataFrame:
    """
    Read RGI6 outlines for the given region(s).

    Arguments
    ---------
    regions: One or multiple RGI regions to read outlines from.
    data_path: The path (relative to the data directory) of the RGI outlines.

    Returns
    -------
    A concatenated GeoDataFrame of all specified RGI regions.
    """
    full_data_path = CONSTANTS.data_path.joinpath(data_path)

    # Convert the region argument to a list if only one region is given
    if not isinstance(regions, list):
        regions = [regions]

    region_data = []
    for region in regions:
        matching = list(full_data_path.glob(f"nsidc0770_{str(region).zfill(2)}_*"))
        if len(matching) == 0:
            raise ValueError(f"No file found for RGI region {region} in {data_path}")

        filepath = matching[0]
        inner_filename = filepath.stem.replace("nsidc0770_", "") + ".shp"
        region_data.append(gpd.read_file(f"/vsizip/{filepath}/{inner_filename}"))

    return pd.concat(region_data)


def read_all_rgi6(data_path: str = "rgi/rgi6") -> gpd.GeoDataFrame:
    """
    Read all RGI6 outlines.

    Arguments
    ---------
    data_path: The path (relative to the data directory) of the RGI outlines.

    Returns
    -------
    A GeoDataFrame of all RGI6 outlines.
    """
    cache_path = surgedetection.cache.get_cache_name("rgi6", [data_path]).with_suffix(".feather")
    if cache_path.is_file():
        return gpd.read_feather(cache_path)
    rgi = pd.concat(read_rgi6(region, data_path=data_path) for region in range(1, 20))

    rgi.to_feather(cache_path)
    return rgi


def read_rgi6_regions(data_path: str = "rgi/rgi6/nsidc0770_00_rgi60_regions.zip") -> gpd.GeoDataFrame:
    """
    Read the predefined RGI6 (O2) regions.

    Arguments
    ---------
    data_path: The path (relative to the data directory) to the RGI regions zipfile.

    Returns
    -------
    A GeoDataFrame of all RGI6 region outlines.
    """
    full_filepath = CONSTANTS.data_path.joinpath(data_path)

    filename = full_filepath.stem.replace("nsidc0770_", "").replace("regions", "O2Regions") + ".shp"
    return gpd.read_file(f"/vsizip/{full_filepath}/{filename}")
