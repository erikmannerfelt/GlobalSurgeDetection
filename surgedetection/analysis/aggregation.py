import concurrent.futures
import os
import shutil
import tempfile
import threading
import time
import warnings
from pathlib import Path

import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.morphology
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from tqdm import tqdm
from tqdm.dask import TqdmCallback

NS_IN_A_YEAR = np.timedelta64(1, "Y").astype("timedelta64[ns]").astype(float)

import surgedetection.cache
import surgedetection.main
import surgedetection.utilities
from surgedetection.constants import CONSTANTS


def interpret_stack(region_id: str = "REGN79E021X24Y05", force_redo: bool = False) -> xr.Dataset:

    cache_filepath = surgedetection.cache.get_cache_name(f"interpret_stack-{region_id}", extension=".zarr")
    if (not force_redo) and cache_filepath.is_dir():
        return xr.open_zarr(cache_filepath)

    warnings.simplefilter("error")

    # Create or load the raw region stack
    stack = surgedetection.main.make_region_stack(region_id)

    # Format the output attrs. For some reason, they disappear below, so they have to be reintroduced
    attrs = stack.attrs.copy()
    if "creation_date" in attrs:
        attrs["source_creation_date"] = attrs["creation_date"]
    attrs["creation_date"] = surgedetection.utilities.now_str()
    var_attrs = {v: stack[v].attrs.copy() for v in stack.data_vars}

    # Subsetting for testing purposes
    if False:
        locations = {"scheele": "RGI60-07.00283", "monaco": "RGI60-07.01494"}
        rgi_id = locations["monaco"]
        bounds = stack["bboxes"].sel(rgi_id=rgi_id).values.ravel()
        # Temporary to try things out
        stack = stack.sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[3], bounds[1]))

    # Average all variables by year. All as of 2022-12-24 consist of just one sample per year
    # so this in reality is just a simpliciation of the time dimension.
    vars_with_time = [v for v in stack.data_vars if "time" in stack[v].dims]
    stack[vars_with_time] = stack[vars_with_time].groupby("time.year").mean().squeeze()

    # Remove the source dimension by averaging for each x/y/year step
    for variable in [v for v in stack.data_vars if "source" in stack[v].dims]:
        stack[variable] = stack[variable].mean("source")

    # Source and time (replaced by year) are now empty, so they can be dropped
    stack = stack.drop_dims(["source", "time"])

    # Generate a glacier mask (True on glaciers)
    stack["glacier_mask"] = stack["rgi_rasterized"] > 0

    # Generate a margin mask by extracting the intersection between the glacier mask and the 
    # an eroded glacier mask (or rather dilated non-glacier mask)
    stack["margin_mask"] = (
        stack["glacier_mask"] & skimage.morphology.binary_dilation(~stack["glacier_mask"], footprint=np.ones((3, 3)))
    ).compute()

    # The DEM and rgi_rasterized arrays need to be loaded for the quantile calculation either way, so this
    # simplifies the syntax slightly. TODO: Make sure this is actually true
    stack["dem"].load()
    stack["rgi_rasterized"].load()
    # Calculate quantiles of elevation for each rgi index
    # TODO: Speed it up by not calculating the quantile for the non-glacier value (0)
    # Problem is then the .sel call below fails because 0 is missing. I tried adding method="nearest" to .sel, but that
    # is slower than just calculating the non-glacier quantile...
    for_quantiles = stack[["dem", "rgi_rasterized"]].stack(xy={"x", "y"}).swap_dims(xy="rgi_rasterized")
    quantiles = (
        for_quantiles["dem"]
        .groupby("rgi_rasterized")
        .quantile([0.33, 0.67])
        .sel(rgi_rasterized=stack["rgi_rasterized"])
    )

    # Generate a terminus mask; True for all locations in the lower third of each individual glacier
    stack["terminus_mask"] = stack["glacier_mask"] & (stack["dem"] <= quantiles.sel(quantile=0.33)).reset_coords(
        drop=True
    )

    # Generate a terminus mask; True for all locations in the upper third of each individual glacier
    stack["accumulation_mask"] = stack["glacier_mask"] & (stack["dem"] >= quantiles.sel(quantile=0.67)).reset_coords(
        drop=True
    )

    # Assign parameters for the output file
    chunks = {"x": 256, "y": 256, "year": 1}
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    save_params = {v: {"compressor": compressor} for v in stack.variables}

    if cache_filepath.is_dir():
        shutil.rmtree(cache_filepath)

    # Reintroduce the attrs that were mysteriously lost along the way
    stack.attrs = attrs
    for variable in var_attrs:
        stack[variable].attrs = var_attrs[variable]

    # Calculate the array so far. Next up, only vars are added, so they can safely be appended
    task = stack.chunk(chunks).to_zarr(cache_filepath, compute=False, encoding=save_params)
    with TqdmCallback(desc=f"A: Computing region {attrs['label']}"):
        task.compute()

    # Make sure everything is loaded from disk now instead
    stack = xr.open_zarr(cache_filepath)
    # Calculate acceleration columns (technically velocity, but it's velocities of velocities!)
    for key in tqdm(["ice_velocity", "dhdt"], desc="B: Calculating acceleration variables"):
        new_var_name = f"{key}2"
        new_err_name = f"{new_var_name}_err"

        # The full data will be riddled with empty time frames, which only creates nans in the diff
        # So, only the slices with data are used to calculate accelerations.
        subset = stack[[key, f"{key}_err"]].dropna("year", "all")

        time_diffs = subset["year"].diff("year")

        new_vars = xr.merge(
            [
                (subset[key].diff("year").chunk(year=30) / time_diffs)
                .reindex({"year": stack["year"]})
                .rename(new_var_name)
                .astype(stack[key].dtype),
                (
                    (
                        (subset[f"{key}_err"].isel(year=slice(1, None)) ** 2)
                        + (subset[f"{key}_err"].isel(year=slice(-1)).values ** 2) ** 0.5
                    )
                    / time_diffs
                )
                .reindex({"year": stack["year"]})
                .rename(new_err_name)
                .astype(stack[f"{key}_err"].dtype),
            ]
        )
        # It seems like each to_zarr call removes the top-level attributes. This reintroduces them each time
        new_vars.attrs = stack.attrs.copy()

        # Copy the attrs of the velocity to the acceleration column
        for old_key, new_key in [(key, new_var_name), (f"{key}_err", new_err_name)]:
            new_vars[new_key].attrs = subset[old_key].attrs.copy()

        save_params = {v: {"compressor": compressor} for v in [new_var_name, new_err_name]}
        new_vars.chunk(chunks).to_zarr(cache_filepath, mode="a", encoding=save_params)

    surgedetection.cache.symlink_to_output(cache_filepath, f"region_stacks/{attrs['label']}")

    return xr.open_zarr(cache_filepath)


def aggregate_region(region_id: str = "REGN79E021X24Y05", force_redo: bool = False, save_chunk_size: int = 100):

    cache_path = surgedetection.cache.get_cache_name(f"aggregate_region-{region_id}", extension=".nc")

    if (not force_redo) and cache_path.is_file():
        return xr.open_dataset(cache_path)
        # return xr.open_zarr(cache_path)
    stack = interpret_stack(region_id=region_id)

    attrs = stack.attrs.copy()
    if "creation_date" in attrs:
        attrs["source_creation_date"] = attrs["creation_date"]

    rgi_ids = stack["rgi_index"].to_series().sort_index()
    rgi_ids = pd.Series(rgi_ids.index.values, index=rgi_ids.values).astype("string")
    # rgi_ids.name = "rgi_rasterized"

    warnings.simplefilter("error")
    xr.set_options(display_max_rows=25)

    if False:
        locations = {"scheele": "RGI60-07.00283", "monaco": "RGI60-07.01494"}
        rgi_id = locations["monaco"]
        bounds = stack["bboxes"].sel(rgi_id=rgi_id).values.ravel()
        # Temporary to try things out
        stack = stack.sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[3], bounds[1]))

    stack = stack.drop_vars(["rgi_id", "rgi_index", "bboxes", "coordinate"])

    stack = stack.stack(xy=["x", "y"]).swap_dims({"xy": "rgi_rasterized"}).reset_coords(drop=True)
    stack = stack.where(stack["rgi_rasterized"] > 0, drop=True)

    err_cols = [v for v in stack.variables if "_err" in v]
    count_cols = [v.replace("_err", "_count") for v in err_cols]
    scopes = [v for v in stack.variables if "_mask" in v]

    if cache_path.is_file():
        os.remove(cache_path)
        # shutil.rmtree(cache_path)

    n = np.unique(stack["rgi_rasterized"].values).shape[0]
    n_stacks = int(np.ceil(n / save_chunk_size))
    first = True
    # compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    aggs = []
    with tqdm(
        total=n_stacks, desc=f"Processing glaciers in chunks of {save_chunk_size}"
    ) as progress_bar, tempfile.TemporaryDirectory() as temp_dir:

        temp_filename = Path(temp_dir).joinpath("arr.zarr")
        for i, (rgi_index, per_rgi) in enumerate(stack.groupby("rgi_rasterized")):
            per_scope = []
            # per_rgi = per_rgi.compute()  # For some reason, this makes it all faster
            for scope in scopes:
                filtered = per_rgi.where(per_rgi[scope])
                scoped_agg = filtered.mean("rgi_rasterized").drop_vars(scopes)
                pixel_counts = filtered[err_cols].notnull().sum("rgi_rasterized")
                scoped_agg[count_cols] = pixel_counts.rename_vars(dict(zip(err_cols, count_cols)))

                per_scope.append(scoped_agg.expand_dims({"scope": np.array([scope], dtype=str)}))

            aggs.append(
                xr.concat(per_scope, "scope").expand_dims({"rgi_id": np.array([rgi_ids[rgi_index]], dtype=str)})
            )
            if len(aggs) >= save_chunk_size or i == (n - 1):
                # save_params = {v: {"compressor": compressor} for v in aggs[0].data_vars}
                if first:
                    zarr_args = {}  # {"encoding": save_params}
                else:
                    zarr_args = {"mode": "a", "append_dim": "rgi_id"}
                xr.concat(aggs, "rgi_id").to_zarr(temp_filename, **zarr_args)
                progress_bar.update()
                aggs.clear()
                first = False

        output = xr.open_zarr(temp_filename)

        for variable in output.data_vars:
            output[variable].encoding.update(zlib=True, complevel=5)

        attrs["creation_date"] = surgedetection.utilities.now_str()
        output.attrs = attrs.copy()

        output.to_netcdf(cache_path)

    surgedetection.cache.symlink_to_output(cache_path, f"region_aggregates/{attrs['label']}")

    return xr.open_dataset(cache_path)
