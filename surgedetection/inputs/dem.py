import functools
import os
import urllib
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio as rio
import shapely.geometry
from pyproj import CRS
from tqdm import tqdm

import surgedetection.cache
import surgedetection.rasters
from surgedetection.constants import CONSTANTS
from surgedetection.rasters import RasterInput

BASE_DOWNLOAD_URL = "https://opentopography.s3.sdsc.edu/raster/COP90/COP90_hh/"
DATA_DIR = CONSTANTS.data_path.joinpath("dem")


def load_dem(
    crs: int | CRS, url: str = "/vsicurl/https://schyttholmlund.com/share/COP90_hh_merged_higher_z.tif"
) -> list[RasterInput]:

    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)

    cache_path = surgedetection.cache.get_cache_name("load_dem", [crs, url], extension="vrt")

    if crs.to_epsg() != 32633:
        raise NotImplementedError("The DEM is currently hardcoded to Svalbard")
    # surgedetection.rasters.create_warped_vrt(url, cache_path, crs.to_wkt())
    cache_path = Path("data/dem/COP90-Svalbard.tif")

    return [
        surgedetection.rasters.RasterInput(
            sources="copernicus-glo90",
            start_dates=pd.Timestamp("2011-01-01"),
            end_dates=pd.Timestamp("2015-07-01"),
            kinds="dem",
            region="07",
            filepath=cache_path,
            multi_source=False,
            multi_date=False,
        )
    ]


def load_region_dem(region: pd.Series) -> RasterInput:

    cache_path = surgedetection.cache.get_cache_name(
        f"load_region_dem-{region['region_id']}", args=[region["geometry"].wkt], extension="vrt"
    )

    if not cache_path.is_file():
        tiles = dem_filenames_from_polygon(region["geometry"])

        filepaths = download_tiles(tiles, progress=True)

        vrt_cache_file = surgedetection.cache.get_cache_name("load_region_dem_vrt", args=filepaths, extension="vrt")
        surgedetection.rasters.build_vrt(filepaths, vrt_cache_file)

        surgedetection.rasters.create_warped_vrt(vrt_cache_file, cache_path, region["crs"].to_wkt())

    return surgedetection.rasters.RasterInput(
        sources="copernicus-glo90",
        start_dates=pd.Timestamp("2011-01-01"),
        end_dates=pd.Timestamp("2015-07-01"),
        kinds="dem",
        region=region["region_id"],
        filepath=cache_path,
        multi_date=False,
        multi_source=False,
    )


def download_tiles(tiles: list[str], progress: bool = True) -> list[Path]:
    tile_dir = DATA_DIR.joinpath("tiles/")

    os.makedirs(tile_dir, exist_ok=True)
    filepaths = []
    downloads = []
    for tile in tiles:
        filepath = tile_dir.joinpath(tile)
        if filepath.is_file():
            filepaths.append(filepath)
        else:
            downloads.append({"url": urllib.parse.urljoin(BASE_DOWNLOAD_URL, tile), "filepath": filepath})

    for query in tqdm(downloads, desc="Downloading DEM tiles", disable=(not progress or (len(downloads) == 0))):
        try:
            surgedetection.utilities.download_file(**query, progress=False)
        except ValueError as exception:
            if "specified key does not exist" in str(exception):
                continue
            raise exception

        # Validate that it's a proper raster
        rio.open(query["filepath"]).close()
        filepaths.append(query["filepath"])

    return filepaths


def dem_filenames_from_polygon(polygon: shapely.geometry.Polygon) -> list[str]:
    """ """
    tiles = dem_tiles()
    filtered = tiles[tiles.intersects(polygon)]

    return filtered["filename"].tolist()


@functools.lru_cache
def dem_tiles() -> gpd.GeoDataFrame:

    tiles = []
    for lon in range(-180, 180):
        for lat in range(-90, 84):
            tiles.append(
                {
                    "geometry": shapely.geometry.box(lon, lat, lon + 1, lat + 1),
                    "filename": dem_filename(longitude=lon, latitude=lat),
                }
            )

    return gpd.GeoDataFrame(tiles, crs=4326)


def dem_filename(longitude: float, latitude: float) -> str:
    """

    Examples
    --------
    >>> dem_filename(longitude=15., latitude=78.)
    'Copernicus_DSM_COG_30_N78_00_E015_00_DEM.tif'
    >>> dem_filename(longitude=-0.5, latitude=0.)
    'Copernicus_DSM_COG_30_N00_00_W001_00_DEM.tif'
    >>> dem_filename(longitude=75.5, latitude=-80.)
    'Copernicus_DSM_COG_30_S80_00_E075_00_DEM.tif'
    """
    lon_ref = "W" if longitude < 0 else "E"
    lat_ref = "S" if latitude < 0 else "N"

    if -1 < longitude < 0:
        longitude = -1
    if -1 < latitude < 0:
        latitude = -1

    lon = str(abs(int(longitude))).zfill(3)
    lat = str(abs(int(latitude))).zfill(2)

    return f"Copernicus_DSM_COG_30_{lat_ref}{lat}_00_{lon_ref}{lon}_00_DEM.tif"
