import hashlib
import os
import tempfile
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features
import rasterio.warp
import xarray as xr
import zarr
from tqdm import tqdm
from tqdm.dask import TqdmCallback

import surgedetection.cache
import surgedetection.inputs
import surgedetection.inputs.aster
import surgedetection.inputs.rgi
import surgedetection.regions
import surgedetection.utilities
from surgedetection.constants import CONSTANTS
from surgedetection.rasters import RasterInput, RasterParams

RASTER_DESCRIPTIONS = {
    "dhdt": "dHdt-1 averages over five-year intervals. The interval is represented by (date - 5yrs, date]",
    "dhdt_err": "dHdt-1 errors for the 'dhdt' arrays of the same date",
    "ice_velocity": "Ice surface velocities averaged per year. The exact date is in the 'ice_velocity_date' variable",
    "ice_velocity_err": "Ice surface velocity errors for the 'ice_velocity' arrays of the same date.",
    "ice_velocity_date": "Ice surface velocity dates expressed in days.",
    "sar_backscatter": "97.5th percentile-reduced backscatter over the winter. The date is the 'end date' of the interval.",
    "sar_backscatter_diff": "Changes in backscatter strength (dB) from one year to the other.",
    "dem": "Elevation values from the Copernicus COP-90 global DEM.",
    "rgi_rasterized": "A rasterized version of the RGI6 outlines. 0 means 'no glacier'. See 'rgi_index' for a mapping to rgi_ids",
    "rgi_index": "A mapping from rgi_ids to the integer indices of 'rgi_rasterized'",
}


def process_raster(
    raster_input: RasterInput,
    raster_params: RasterParams,
    temp_dir_filepath: Path,
    n_threads: int | None = None,
    progress_bar: tqdm | None = None,
) -> Path | None:
    """
    Process (warp and prepare metadata for) one raster input and return a path the temporary nc.

    Arguments
    ---------
    raster_input: One RasterInput class (see the .rasters module)

    Returns
    -------
    A path to the netcdf.
    """
    with rio.open(raster_input.filepath) as raster:

        window = raster.window(*raster_params.bounding_box())
        src_transform = raster.window_transform(window)
        src_crs = raster.crs
        src_dtype = raster.dtypes[0]

        orig = raster.read(1, window=window, masked=True)

    orig = orig.astype("float32").filled(np.nan)

    if np.count_nonzero(np.isfinite(orig)) == 0:
        return None

    if all(
        (
            src_transform == raster_params.transform,
            src_crs == raster_params.crs,
            orig.shape == raster_params.shape,
        )
    ):
        array = orig
    else:
        array = np.empty(shape=raster_params.shape(), dtype=src_dtype)
        rasterio.warp.reproject(
            orig,
            array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=raster_params.transform,
            dst_crs=raster_params.crs,
            dst_resolution=CONSTANTS.pixel_size,
            resampling=rasterio.warp.Resampling.cubic_spline,
            num_threads=n_threads or ((os.cpu_count() or 2) - 1),
        )
        del orig

    filename = temp_dir_filepath.joinpath(hashlib.sha1(str(raster_input.filepath).encode()).hexdigest()).with_suffix(
        ".nc"
    )

    coords = raster_params.xarray_coords()
    if raster_input.multi_date:
        coords.append(("time", [raster_input.end_date]))

    if raster_input.multi_source:
        coords.append(("source", [raster_input.source]))

    arr = xr.DataArray(
        array.reshape(array.shape + (1,) * (len(coords) - len(array.shape))),
        coords=coords,
        name=raster_input.kind,
        attrs={
            "description": RASTER_DESCRIPTIONS[str(raster_input.kind)],
            "interval_seconds": (raster_input.end_date - raster_input.start_date).total_seconds(),  # type: ignore
            "source": "variable" if raster_input.multi_source else raster_input.source,
            "start_date": "variable" if raster_input.multi_date else raster_input.start_date.isoformat(),
            "end_date": "variable" if raster_input.multi_date else raster_input.end_date.isoformat(),
        },
    )

    arr.to_netcdf(filename)

    progress_bar.update()

    return filename


def make_region_stack(
    region_id: str = "REGN79E021X24Y05", n_threads: int | None = None, force_redo: bool = False
) -> xr.Dataset:

    # Load the associated region id
    region = surgedetection.regions.make_glacier_regions().query(f"region_id == '{region_id}'").iloc[0]

    cache_path = surgedetection.cache.get_cache_name(f"region_stack-{region_id}").with_suffix(".zarr")

    if cache_path.is_dir() and not force_redo:
        return xr.open_zarr(cache_path)

    # Define output parameters of the outgoing rasters
    raster_params = RasterParams.from_bounds(
        bounding_box=[region["xmin_proj"], region["ymin_proj"], region["xmax_proj"], region["ymax_proj"]],
        height=region["height_px"],
        width=region["width_px"],
        crs=region["crs"],
    )
    # Read RGI outlines which will be rasterized
    rgi = surgedetection.inputs.rgi.read_rgi6(region["O1_regions"])
    rgi = rgi[rgi.intersects(region.geometry)].reset_index().to_crs(raster_params.crs)
    # Index 0 will be "no glacier", so all of them are incremented by 1
    rgi.index += 1

    spatial_coords = raster_params.xarray_coords()
    # Define the outgoing base-level attributes of the dataset
    attrs: dict[Hashable, Any] = {"bounding_box": raster_params.bounding_box(),} | region[
        ["name", "region_id", "label", "rgi_regions", "n_glaciers", "glacier_area", "height_px", "width_px", "crs_epsg"]
    ].to_dict()


    # The base dataset contains everything except the rasters which are iteratively appended later.
    base = xr.Dataset(
        {
            # The rasterized version of the RGI. 0 is "no glacier", 1 is the first, etc.
            "rgi_rasterized": xr.DataArray(
                rasterio.features.rasterize(
                    zip(rgi.geometry, rgi.index),
                    out_shape=raster_params.shape(),
                    transform=raster_params.transform,
                ),
                coords=spatial_coords,
                name="rgi_mask",
            ),
            # A mapping from the rgi_rasterized indices to rgi_ids
            "rgi_index": xr.DataArray(rgi.index, coords=[("rgi_id", rgi["RGIId"])]),
        }
    )

    # Add rounded bounding boxes for each glacier, for simpler fast evaluation of single glaciers
    bboxes = np.array(rgi.geometry.apply(lambda r: r.bounds).values.tolist())
    for i in range(4):
        res = raster_params.xres() if i % 2 == 0 else raster_params.yres()
        bboxes[:, i] -= bboxes[:, i] % res
        if i >= 2:
            bboxes[:, i] += res
    base["bboxes"] = xr.DataArray(bboxes, coords=[base["rgi_id"], ("coordinate", ["xmin", "ymin", "xmax", "ymax"])])

    with tempfile.TemporaryDirectory() as temp_dir_str:
        # temp_dir = tempfile.TemporaryDirectory()
        temp_dir_filepath = Path(temp_dir_str)

        # Start making a filepaths list containing all separate nc files (to combine further down in open_mfdataset)
        filepaths = [temp_dir_filepath.joinpath("base.nc")]

        # Save the base dataset and remove it from memory. It's not needed for a while
        base.to_netcdf(filepaths[0])
        del base

        # Read all raster inputs, which will iteratively be converted to nc's with correct bounds
        raster_inputs = surgedetection.inputs.get_all_rasters(crs=region["crs"])
        #raster_inputs = raster_inputs[:5]

        with tqdm(total=len(raster_inputs), desc="Reprojecting datasets") as progress_bar:
            for raster_input in raster_inputs:
                result = process_raster(
                    raster_input=raster_input,
                    raster_params=raster_params,
                    temp_dir_filepath=temp_dir_filepath,
                    n_threads=n_threads,
                    progress_bar=progress_bar,
                )
                if result is not None:
                    filepaths.append(result)

        chunks = {"x": 256, "y": 256, "time": 1}

        with dask.config.set({"array.slicing.split_large_chunks": True}):
            data = xr.open_mfdataset(filepaths, chunks="auto", parallel=True)
            # Avoid complaining about this being an object array
            for coord in ["source", "rgi_id"]:
                if coord in data.coords:
                    data.coords[coord] = data.coords[coord].astype(str)
            data.attrs = attrs

            # For each temporal variable, it will be combined with all other temporal variables along the time
            # dimension. To quickly subset one variable, one must then have to load it to filter out which time
            # coordinates are present. This time dict allows setting what time coords exist for each variable
            # as an attr, which can speed up reading it later.
            for variable in [v for v in data.data_vars if "time" in data[v].dims]:
                data[variable].attrs["times"] = np.unique(data[variable]["time"].dropna("time").values)

            for variable in RASTER_DESCRIPTIONS:
                if variable not in data.data_vars:
                    continue
                data[variable].attrs["description"] = RASTER_DESCRIPTIONS[variable]

            for variable in data.data_vars:
                if not all(d in data[variable].dims for d in ("x", "y")):
                    continue

                data[variable].attrs.update(
                    {
                        "GDAL_AREA_OR_POINT": "Area",
                        "_CRS": {"wkt": str(region["crs"].to_wkt())},
                    }
                )

            for dim in ["x", "y"]:
                data[dim].attrs = {
                    "units": "m",
                    "standard_name": f"projection_{dim}_coordinate",
                    "long_name": "Easting" if dim == "x" else "Northing",
                }

            data.attrs["creation_date"] = surgedetection.utilities.now_str()

            compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)

            task = data.chunk(chunks).to_zarr(
                cache_path, mode="w", encoding={v: {"compressor": compressor} for v in data.variables}, compute=False
            )
            with TqdmCallback(desc=f"Writing region {region['label']}"):
                task.compute()

        del data

    surgedetection.cache.symlink_to_output(cache_path, f"raw_region_stacks/{region['label']}")

    return xr.open_zarr(cache_path)
