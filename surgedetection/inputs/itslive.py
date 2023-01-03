import urllib
import warnings
from pathlib import Path
import asyncio

import pandas as pd
import rasterio as rio
import rasterio.warp
import xarray as xr
from pyproj import CRS
import pyproj
from tqdm import tqdm
from tqdm.dask import TqdmCallback
import geopandas as gpd
from io import StringIO, BytesIO
import requests
import numpy as np
import zarr

import surgedetection.cache
import surgedetection.rasters
import surgedetection.regions
import surgedetection.utilities
from surgedetection.constants import CONSTANTS
from surgedetection.rasters import RasterInput

DATA_DIR = CONSTANTS.data_path.joinpath("its-live")
BASE_DOWNLOAD_URL = "https://its-live-data.s3.amazonaws.com/velocity_mosaic/landsat/v00.0/annual/"
TILE_METADATA_URL = "https://its-live-data.s3.amazonaws.com/datacubes/catalog_v02.json"
YEARS = list(range(1985, 2019))


def get_files(region: pd.Series) -> list[RasterInput]:

    rgi_regions = list(map(lambda s: int(s.split("-")[0]), region["rgi_regions"].split("/")))
    filepaths = download_all_itslive(rgi_regions=rgi_regions)

    rasters = []
    variables = {"v": "ice_velocity", "v_err": "ice_velocity_err", "date": "ice_velocity_date"}
    for filepath in filepaths:

        itslive_region = filepath.stem.split("_")[0]
        year = int(filepath.stem.split("_")[-1])
        start_date = pd.Timestamp(year=year, month=1, day=1)
        end_date = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)

        variable_paths = [f"NETCDF:{filepath}:{variable}" for variable in variables]

        cache_path = surgedetection.cache.get_cache_name(
            f"itslive-read_files-{itslive_region}-{year}", [variables, DATA_DIR]
        ).with_suffix(".vrt")

        if not cache_path.is_file():
            surgedetection.rasters.build_vrt(variable_paths, cache_path, gdal_kwargs={"separate": True})

        rasters.append(
            RasterInput(
                sources="its-live",
                start_dates=start_date,
                end_dates=end_date,
                kinds=list(variables.values()),
                region=itslive_region,
                filepath=cache_path,
                multi_source=False,
                multi_date=True,
            )
        )

    return rasters


def rgi_mapping(rgi_region: int | str, res: str = "_G0240") -> str | None:
    if isinstance(rgi_region, str):
        rgi_region = int(rgi_region)

    match rgi_region:
        case 1 | 2:
            return "ALA" + res
        case 3:
            return "CAN" + res
        case 5:
            return "GRE" + res
        case 6:
            return "ICE" + res
        case 7 | 9:
            return "SRA" + res
        case 13 | 14 | 15:
            return "HMA" + res
        case 17:
            return "PAT" + res
        case 19:
            return "ANT" + res

    return None


def download_all_itslive(rgi_regions: list[int] | None = None, progress: bool = True) -> list[Path]:
    labels = set(filter(None, map(rgi_mapping, rgi_regions or CONSTANTS.rgi_regions)))

    filepaths = []
    queries = []
    for label in labels:
        for year in YEARS:
            filename = f"{label}_{year}.nc"
            filepath = DATA_DIR.joinpath(filename)
            if filepath.is_file():
                filepaths.append(filepath)
            else:
                queries.append({"url": urllib.parse.urljoin(BASE_DOWNLOAD_URL, filename), "filepath": filepath})

    for query in tqdm(queries, desc="Downloading ITS-LIVE data", disable=(not progress or len(queries) == 0)):
        filepath = surgedetection.utilities.download_file(**query, progress=False)

        # Validate that the filepath is a valid nc file
        xr.open_dataset(filepath).close()
        filepaths.append(filepath)

    return filepaths

def process_tile(filename: Path = Path("vel_tile0_svalbard.zarr"), region: pd.Series | str = "REGN79E021X24Y05"):

    if isinstance(region, str):
        region = surgedetection.regions.make_glacier_regions().query(f"region_id == '{region}'").iloc[0]
    data = xr.open_zarr(filename)#.isel(mid_date=slice(-2, -1))
    attrs = data.attrs.copy()

    src_crs = CRS.from_epsg(data.attrs["projection"])
    data = data.groupby("mid_date.year").map(lambda df: df.weighted((1 / df["v_error"]).fillna(0)).mean("mid_date"))
    #data = data.sel(year=slice(2018, None))

    src_xres = float(data["x"].diff("x").isel(x=0))
    src_yres = float(data["y"].diff("y").isel(y=0))

    src_transform = rio.transform.from_origin(
        west=float(data["x"].min()) - src_xres / 2,
        north=float(data["y"].max()) + src_yres / 2,
        xsize=src_xres, 
        ysize=src_yres
    )
    dst_transform = rio.transform.from_bounds(*region[["xmin_proj", "ymin_proj", "xmax_proj", "ymax_proj", "width_px", "height_px"]])
    dst_crs = region["crs"]
    out_params = surgedetection.rasters.RasterParams(transform=dst_transform, width=region["width_px"], height=region["height_px"], crs=dst_crs)

    out_shape = (data["v"].shape[0], out_params.height, out_params.width)

    names = {
        "v": "ice_velocity",
        "v_error": "ice_velocity_err",
    }

    with TqdmCallback(desc="Making yearly mosaics.", smoothing=0):
        data = data.compute()
    arrs = {}
    for variable in ["v", "v_error"]:
        destination = np.empty(out_shape, dtype="float32") + np.nan

        src_arr = data[variable].to_numpy()
        rasterio.warp.reproject(
            src_arr,
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.cubic_spline,
        )
        arrs[names[variable]] = (xr.DataArray(
            destination,
            coords=[data["year"]] + out_params.xarray_coords(),
            name=names[variable]
        ))

    out = xr.Dataset(arrs)
    out.attrs = attrs

    out.to_netcdf("out.nc")

    print(out)

def process_raw_tiles(data: xr.Dataset,out_filepath: Path, region: pd.Series | str = "REGN79E021X24Y05"):
    if isinstance(region, str):
        region = surgedetection.regions.make_glacier_regions().query(f"region_id == '{region}'").iloc[0]

    attrs = data.attrs.copy()

    src_crs = CRS.from_epsg(data.attrs["projection"])
    #data = data.groupby("mid_date.year").map(lambda df: df.weighted((1 / df["v_error"]).fillna(0)).mean("mid_date"))

    src_xres = float(data["x"].diff("x").isel(x=0))
    src_yres = float(data["y"].diff("y").isel(y=0))

    src_transform = rio.transform.from_origin(
        west=float(data["x"].min()) - src_xres / 2,
        north=float(data["y"].max()) + src_yres / 2,
        xsize=src_xres, 
        ysize=src_yres
    )
    dst_transform = rio.transform.from_bounds(*region[["xmin_proj", "ymin_proj", "xmax_proj", "ymax_proj", "width_px", "height_px"]])
    dst_crs = region["crs"]
    out_params = surgedetection.rasters.RasterParams(transform=dst_transform, width=region["width_px"], height=region["height_px"], crs=dst_crs)

    out_shape = (data["v"].shape[0], out_params.height, out_params.width)

    names = {
        "v": "ice_velocity",
        "v_error": "ice_velocity_err",
    }

    with TqdmCallback(desc="Making yearly mosaics.", smoothing=0):
        data = data.compute()
    arrs = {}
    for variable in ["v", "v_error"]:
        destination = np.empty(out_shape, dtype="float32") + np.nan

        src_arr = data[variable].to_numpy()
        rasterio.warp.reproject(
            src_arr,
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.cubic_spline,
        )
        arrs[names[variable]] = (xr.DataArray(
            destination,
            coords=[data["year"]] + out_params.xarray_coords(),
            name=names[variable]
        ))

    out = xr.Dataset(arrs)
    out.attrs = attrs

    out.to_zarr(out_filepath, encoding={v: {"compressor": zarr.Blosc(cname="zstd", clevel=3, shuffle=2)} for v in out.data_vars})

    print(out)

def get_tiles(region: pd.Series | str = "REGN79E021X24Y05"):

    if isinstance(region, str):
        region = surgedetection.regions.make_glacier_regions().query(f"region_id == '{region}'").iloc[0]

    tile_meta = get_tile_metadata()
    tile_meta = tile_meta[tile_meta.intersects(region.geometry)]

    if tile_meta.shape[0] == 0:
        return []

    files = []
    for epsg, tiles in tile_meta.groupby("epsg"):

        cache_path = surgedetection.cache.get_cache_name(f"itslive-get_tiles/yearly_mosaics-{region['region_id']}", args=tiles["zarr_url"], extension="zarr")

        if cache_path.is_dir():
            files.append(cache_path)
            
        data = []
        for tile in tqdm(asyncio.run(get_multiple_zarrs(tiles["zarr_url"].values[:2])).result()):
            tile2 = tile.groupby("mid_date.year").map(lambda df: df.weighted(df["v_error"].max() * xr.apply_ufunc(np.reciprocal, df["v_error"], dask="allowed")).mean("mid_date"))
            tile2.attrs = tile.attrs.copy()
            data.append(tile2)
            

        process_raw_tiles(xr.merge(data), out_filepath=cache_path, region=region)
        files.append(cache_path)


async def get_multiple_zarrs(urls: list[str]) -> list[xr.DataArray]:
    return asyncio.gather(*map(get_zarr,urls)) 

async def get_zarr(url: str) -> xr.DataArray:
    return xr.open_zarr(url)[["v", "v_error"]]
    

def get_tile_metadata() -> gpd.GeoDataFrame:
    version = "2.0"
    cache_path = surgedetection.cache.get_cache_name("itslive-get_tile_metadata", [version], extension=".geojson")

    if cache_path.is_file():
        return gpd.read_file(cache_path)

    response = requests.get(TILE_METADATA_URL, timeout=10)

    if response.status_code != 200:
        raise ValueError(response.status_code, response.content)

    data = gpd.read_file(BytesIO(response.content))

    data.to_file(cache_path, driver="GeoJSON")

    return data
    
