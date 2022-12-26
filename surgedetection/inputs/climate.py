import xarray as xr
from surgedetection.constants import CONSTANTS
from surgedetection.rasters import RasterInput, RasterParams
import surgedetection.rasters
import surgedetection.cache
from pathlib import Path
import pyproj
from pyproj import CRS
import os
import numpy as np
import tempfile
import pandas as pd
import calendar
import rasterio.warp
import rasterio as rio

DATA_DIR = CONSTANTS.data_path.joinpath("climate")


def get_files() -> list[RasterInput]:
    era5_monthly_path = download_era5()
    era5_monthly = xr.open_dataset(era5_monthly_path, chunks={"time": 1})

    labels = {
        "t2m": "air_temperature",
        "sf": "solid_precipitation",
        "tp": "total_precipitation"
    }

    rasters = []
    for variable in era5_monthly.data_vars:
        for i, time in enumerate(era5_monthly.time.values, start=1):
            time_str = time.astype("datetime64[M]").astype(str)
            vrt_path = surgedetection.cache.get_cache_name(f"era5-{variable}-{time_str}", args=[era5_monthly_path.name], extension="vrt")

            if not vrt_path.is_file():
                surgedetection.rasters.separate_band_vrt(f"NETCDF:{era5_monthly_path}:{variable}", vrt_path, [i], gdal_kwargs={"outputSRS": "+proj=longlat +datum=WGS84 +lon_wrap=180", "outputBounds": [0, -90, 360, 90]})

            date = pd.Timestamp(time)

            rasters.append(RasterInput(
                source="era5",
                kind=labels[variable],
                region="global",
                start_date=date,
                end_date=date.replace(day=calendar.monthrange(date.year, date.month)[1], hour=23, minute=23, second=59),
                filepath=vrt_path,
                multi_source=False,
                multi_date=True,
                time_prefix="era"
            ))

    return rasters


def get_era5_vrts(crs: CRS | int) -> tuple[list[np.datetime64], dict[str, Path]]:
    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)
    era5_monthly_path = download_era5()
    era5_monthly = xr.open_dataset(era5_monthly_path, chunks={"time": 1})
    times = era5_monthly["time"].values.astype("datetime64[ns]")

    labels = {
        "t2m": "air_temperature",
        "sf": "solid_precipitation",
        "tp": "total_precipitation"
    }

    vrts = {}
    for variable in era5_monthly.data_vars:
        if variable == "grid_mapping":
            continue
        vrt_path = surgedetection.cache.get_cache_name(f"era5-{variable}", args=[era5_monthly_path.name, crs.to_wkt()], extension="vrt")

        #if not vrt_path.is_file():

            #surgedetection.rasters.build_vrt([f"NETCDF:{era5_monthly_path}:{variable}"], vrt_path, gdal_kwargs={"outputSRS": CRS.from_epsg(4326).to_wkt(), "outputBounds": [10, 50, 35, 80]})

        vrts[labels[variable]] = f"NETCDF:{era5_monthly_path}:{variable}"

    return times.tolist(), vrts

def warp_to_grid(raster_params: RasterParams) -> xr.Dataset:
    from osgeo import gdal

    times, vrts = get_era5_vrts(raster_params.crs)

    coords = [("time", np.array(times, dtype="datetime64[ns]"))] + raster_params.xarray_coords()

    data = {}
        
    with tempfile.TemporaryDirectory() as temp_dir:
        for variable in vrts:
            vrt_path = Path(temp_dir).joinpath(f"{variable}.vrt")
            # TODO: This is how VRTs are properly built! No other way!!
            gdal.Warp(
                str(vrt_path),
                str(vrts[variable]),
                outputBounds=raster_params.bounding_box(),
                format="VRT",
                srcSRS=CRS.from_epsg(4326).to_wkt(),
                dstSRS=raster_params.crs.to_wkt(),
                multithread=True,
                xRes=raster_params.xres(),
                yRes=raster_params.yres(),
                resampleAlg=rio.warp.Resampling.bilinear
            )
            with rio.open(vrt_path) as raster:
                orig = raster.read(masked=True).filled(np.nan)
    
            data[variable] = xr.DataArray(orig, coords=coords, name=variable)

    data = xr.Dataset(data)

    return data

    


def download_era5():
    era5_monthly_query = {
                "variable": ["2m_temperature", "snowfall", "total_precipitation"],
                "pressure_level": "1000",
                "product_type": "monthly_averaged_reanalysis",
                "month": [str(m).zfill(2) for m in range(13)],
                "time": "00:00",
                "year": [str(y) for y in CONSTANTS.era5_years],
                "format": "netcdf",
    }

    # As with the other network queries, they should be in the "data/" dir and not the cache
    cache_path = surgedetection.cache.get_cache_name("era5_monthly", kwargs=era5_monthly_query)
    data_path = DATA_DIR.joinpath(cache_path.stem[:20] + ".nc")


    if not data_path.is_file():
        import cdsapi
        cds = cdsapi.Client()
        os.makedirs(data_path.parent, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_path = temp_dir_path.joinpath("era5.nc")
            cds.retrieve(
                "reanalysis-era5-single-levels-monthly-means",
                era5_monthly_query,
                temp_path,
            )
            orig = xr.open_dataset(temp_path, chunks={"time": 20})
            var_attrs = {v: orig[v].attrs.copy() for v in orig.data_vars}
            attrs = orig.attrs.copy()

            if "expver" in orig.dims:
                orig = orig.mean("expver")
            # The grid mapping of ERA5 data is crazy. TODO: Explain why
            # https://confluence.ecmwf.int/display/CKB/ERA5%3A+What+is+the+spatial+reference
            east = orig.sel(longitude=slice(0, 179.99))
            west = orig.sel(longitude=slice(180, 361))
            west.coords["longitude"] = west.coords["longitude"] - 360

            wgs84 = pyproj.CRS.from_epsg(4326)
            fixed = xr.concat([west, east], dim="longitude").isel(longitude=slice(1, None), latitude=slice(1, -1))
            for variable in fixed.data_vars:
                fixed[variable].encoding.update({"zlib": True, "complevel": 9, "_FillValue": -32767})
                fixed[variable].attrs.update({"grid_mapping": "grid_mapping"} | var_attrs[variable])
            fixed["grid_mapping"] = xr.DataArray([''], attrs={"crs_wkt": wgs84.to_wkt()})
            fixed.attrs.update(attrs)


            data.to_netcdf(data_path)

    return data_path

def fix():
    orig = xr.open_dataset("data/climate/era5_monthly_accf212.nc")


    fixed.to_netcdf("maybe_fixed.nc")
