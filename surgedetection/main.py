import concurrent.futures
import os
import tempfile
import threading
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import dask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features
import rasterio.warp
import shapely.wkb
import xarray as xr
from tqdm import tqdm

import surgedetection.cache
import surgedetection.inputs
import surgedetection.inputs.aster
import surgedetection.inputs.rgi
import surgedetection.regions
from surgedetection.constants import CONSTANTS


def make_glacier_stack(glims_id: str = "G014442E77835N") -> xr.Dataset:

    cache_path = surgedetection.cache.get_cache_name(f"glacier_stack-{glims_id}").with_suffix(".nc")

    if cache_path.is_file():
        return xr.load_dataset(cache_path)

    crs = rio.crs.CRS.from_epsg(32633)
    rgi = (
        gpd.read_file("data/rgi/rgi7/RGI2000-v7.0-G-07_svalbard_jan_mayen/RGI2000-v7.0-G-07_svalbard_jan_mayen.shp")
        .query(f"glims_id == '{glims_id}'")
        .to_crs(crs)
        .iloc[0]
    )

    # Convert the Polygon Z to a normal polygon
    rgi.geometry = shapely.wkb.loads(shapely.wkb.dumps(rgi.geometry, output_dimension=2))

    bounding_box = list(rgi.geometry.bounds)
    mod = 100
    for i in range(len(bounding_box)):
        bounding_box[i] -= bounding_box[i] % mod
        if i <= 1:
            bounding_box[i] -= 30 * mod
        else:
            bounding_box[i] += 30 * mod

    shape = (int((bounding_box[3] - bounding_box[1]) / mod), int((bounding_box[2] - bounding_box[0]) / mod))

    transform = rio.transform.from_bounds(*bounding_box, *(shape[::-1]))
    spatial_coords = [
        ("northing", (np.linspace(bounding_box[1] + mod / 2, bounding_box[3] - mod / 2, shape[0]))[::-1]),
        ("easting", np.linspace(bounding_box[0] + mod / 2, bounding_box[2] - mod / 2, shape[1])),
    ]
    attrs: dict[Hashable, Any] = {
        "crs": crs.to_wkt(),
        "resolution": [mod, mod],
        "height_px": shape[0],
        "width_px": shape[1],
        "glacier_geometry": rgi.geometry.wkt,
        "bounding_box": bounding_box,
    }
    attrs.update({key: value for key, value in rgi.items() if (not pd.isna(value) and key != "geometry")})
    attrs["glacier_geometry"] = rgi.geometry.wkt

    arrays = [
        xr.DataArray(
            rasterio.features.rasterize(
                [rgi.geometry],
                out_shape=shape,
                transform=transform,
            )
            == 1,
            coords=spatial_coords,
            name="rgi_mask",
            attrs={"description": "The mask is true for inliers."},
        )
    ]

    aster = surgedetection.inputs.get_all_paths(crs=crs)
    for index, filepath in aster.items():
        index = dict(zip(aster.index.names, index))

        with rio.open(filepath) as raster:

            window = raster.window(*bounding_box)

            array = np.empty(shape=shape, dtype="float32")
            rasterio.warp.reproject(
                raster.read(1, window=window, masked=True).astype("float32").filled(np.nan),
                array,
                src_transform=raster.window_transform(window),
                src_crs=raster.crs,
                dst_transform=transform,
                dst_crs=crs,
                dst_resolution=mod,
                resampling=rasterio.warp.Resampling.cubic_spline,
            )
            if np.count_nonzero(np.isfinite(array)) == 0:
                continue

            descriptions = {
                "dhdt": "dHdt-1 averages over five-year intervals. The interval is represented by (date - 5yrs, date]",
                "dhdt_err": "dHdt-1 errors for the 'dhdt' arrays of the same date",
                "ice_velocity": "Ice surface velocities averaged per year. The exact date is in the 'ice_velocity_date' variable",
                "ice_velocity_err": "Ice surface velocity errors for the 'ice_velocity' arrays of the same date.",
                "ice_velocity_date": "Ice surface velocity dates expressed in days.",
                "sar_backscatter": "97.5th percentile-reduced backscatter over the winter. The date is the 'end date' of the interval.)",
            }

            arrays.append(
                xr.DataArray(
                    array.reshape(array.shape + (1, 1)),
                    coords=spatial_coords + [("time", [index["end_date"]]), ("source", [index["source"]])],
                    name=index["kind"],
                    attrs={
                        "description": descriptions[index["kind"]],
                        "interval_seconds": (index["end_date"] - index["start_date"]).total_seconds(),
                    },
                )
            )

    print("Merging")

    # arrs: list[xr.DataArray] = []
    # data = xr.Dataset()
    data = xr.merge(arrays)
    # xr.merge is extremely memory-inefficient. Therefore, the merge is split up in chunks
    # for _ in range(len(arrays)):
    #    continue
    #    if len(arrs) == 20:
    #        data = xr.merge([data] + arrs)

    #        arrs.clear()
    #     arrs.append(arrays.pop(0))

    data.attrs.update(attrs)

    print("Saving")
    data.to_netcdf(cache_path)

    return data


def make_region_stack(region_id: str = "REGN79E021X24Y05", n_threads: int | None = None) -> xr.Dataset:

    region = surgedetection.regions.make_glacier_regions().query(f"region_id == '{region_id}'").iloc[0]

    cache_path = surgedetection.cache.get_cache_name(f"region_stack-{region_id}").with_suffix(".nc")

    descriptions = {
        "dhdt": "dHdt-1 averages over five-year intervals. The interval is represented by (date - 5yrs, date]",
        "dhdt_err": "dHdt-1 errors for the 'dhdt' arrays of the same date",
        "ice_velocity": "Ice surface velocities averaged per year. The exact date is in the 'ice_velocity_date' variable",
        "ice_velocity_err": "Ice surface velocity errors for the 'ice_velocity' arrays of the same date.",
        "ice_velocity_date": "Ice surface velocity dates expressed in days.",
        "sar_backscatter": "97.5th percentile-reduced backscatter over the winter. The date is the 'end date' of the interval.)",
        "rgi_mask": "RGI6.",
        "RGIId": "",
    }

    if cache_path.is_file():
        data = xr.open_dataset(cache_path)
        # Avoid complaining about this being an object array
        data["RGIId"] = data["RGIId"].astype(str)
        return data

    shape = (region["height_px"], region["width_px"])
    bounding_box = [region["xmin_proj"], region["ymin_proj"], region["xmax_proj"], region["ymax_proj"]]
    transform = rio.transform.from_bounds(*bounding_box, *(shape[::-1]))

    rgi = surgedetection.inputs.rgi.read_rgi6(region["O1_regions"])
    rgi = rgi[rgi.intersects(region.geometry)].reset_index().to_crs(region["crs"])
    rgi.index += 1

    spatial_coords = [
        (
            "y",
            np.linspace(
                region["ymin_proj"] + CONSTANTS.pixel_size / 2,
                region["ymax_proj"] - CONSTANTS.pixel_size / 2,
                region["height_px"],
            )[::-1],
        ),
        (
            "x",
            np.linspace(
                region["xmin_proj"] + CONSTANTS.pixel_size / 2,
                region["xmax_proj"] - CONSTANTS.pixel_size / 2,
                region["width_px"],
            ),
        ),
    ]
    attrs: dict[Hashable, Any] = {
        "bounding_box": bounding_box,
        "Conventions": "CF-1.5",
        "GDAL_AREA_OR_POINT": "Area",
    } | region[
        ["name", "region_id", "label", "rgi_regions", "n_glaciers", "glacier_area", "height_px", "width_px", "crs_epsg"]
    ].to_dict()

    arrays = [
        xr.DataArray(
            rasterio.features.rasterize(
                zip(rgi.geometry, rgi.index),
                out_shape=shape,
                transform=transform,
            ),
            coords=spatial_coords,
            name="rgi_mask",
            attrs={
                "description": descriptions["rgi_mask"],
                "grid_mapping": "grid_mapping",
            },
        ),
        xr.DataArray(
            b"",
            attrs={
                "crs_wkt": region["crs"].to_wkt(),
                "GeoTransform": f"{region['xmin_proj']:.1f} {CONSTANTS.pixel_size:.1f} 0 {region['ymax_proj']:.1f} 0 -{CONSTANTS.pixel_size:.1f}",
            },
            name="grid_mapping",
        ),
        xr.DataArray(rgi["RGIId"], dims=["glacier_index"], attrs={"description": descriptions["RGIId"]}),
    ]

    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_filepath = Path(temp_dir.name)

    xr.merge(arrays).to_netcdf(temp_dir_filepath.joinpath("base.nc"))
    del arrays
    filepaths = [temp_dir_filepath.joinpath("base.nc")]

    items = surgedetection.inputs.get_all_paths(crs=region["crs"])
    infos = [dict(zip(items.index.names, index)) | {"filepath": filepath} for index, filepath in items.items()]

    with tqdm(total=len(infos), desc="Reprojecting datasets") as progress_bar:

        def process(info: dict[str, object]) -> Path | None:
            with rio.open(info["filepath"]) as raster:

                window = raster.window(*bounding_box)
                src_transform = raster.window_transform(window)
                crs = raster.crs

                orig = raster.read(1, window=window, masked=True)

            orig = orig.astype("float32").filled(np.nan)
            array = np.empty(shape=shape, dtype="float32")
            rasterio.warp.reproject(
                orig,
                array,
                src_transform=src_transform,
                src_crs=crs,
                dst_transform=transform,
                dst_crs=region["crs"],
                dst_resolution=CONSTANTS.pixel_size,
                resampling=rasterio.warp.Resampling.cubic_spline,
                num_threads=n_threads or ((os.cpu_count() or 2) - 1),
            )
            del orig

            if np.count_nonzero(np.isfinite(array)) == 0:
                return None

            filename = Path(temp_dir.name).joinpath(str(abs(hash(str(info))))).with_suffix(".nc")

            xr.DataArray(
                array.reshape(array.shape + (1, 1)),
                coords=spatial_coords + [("time", [info["end_date"]]), ("source", [info["source"]])],
                name=info["kind"],
                attrs={
                    "description": descriptions[str(info["kind"])],
                    "interval_seconds": (info["end_date"] - info["start_date"]).total_seconds(),  # type: ignore
                    "grid_mapping": "grid_mapping",
                },
            ).to_netcdf(filename)
            progress_bar.update()

            return filename

        for info in infos:
            result = process(info)
            if result is not None:
                filepaths.append(result)

    with dask.config.set({"array.slicing.split_large_chunks": True}):
        data = xr.open_mfdataset(filepaths)
        # Avoid complaining about this being an object array
        data["RGIId"] = data["RGIId"].astype(str)
        data.attrs = attrs

        for dim in ["x", "y"]:
            data[dim].attrs = {
                "units": "m",
                "standard_name": f"projection_{dim}_coordinate",
                "long_name": "{dim} coordinate of projection",
            }

        def nodata(dtype: Any) -> Any:
            try:
                np.finfo(dtype)
                return -9999.0
            except ValueError:
                pass
            return np.iinfo(dtype).max

        save_params = {
            v: {"zlib": True, "complevel": 5, "_FillValue": nodata(data[v].dtype)}
            for v in data.variables
            if v not in (["grid_mapping", "RGIId"] + list(data.coords))
        }
        save_params["time"] = {"_FillValue": 0}
        print("Saving")
        data.to_netcdf(cache_path, encoding=save_params)

    surgedetection.cache.symlink_to_output(cache_path, f"region_stacks/{region['label']}")

    return data
