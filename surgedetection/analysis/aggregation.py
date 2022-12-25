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
    chunks = {"x": 256, "y": 256, "year": 3}
    encoding = {"compressor": zarr.Blosc(cname="zstd", clevel=9, shuffle=2)}

    if cache_filepath.is_dir():
        shutil.rmtree(cache_filepath)

    # Reintroduce the attrs that were mysteriously lost along the way
    stack.attrs = attrs
    for variable in var_attrs:
        stack[variable].attrs = var_attrs[variable]

    # Calculate the array so far. Next up, only vars are added, so they can safely be appended
    task = stack.chunk(chunks).to_zarr(cache_filepath, compute=False, encoding={k: encoding for k in stack.variables})
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

        with warnings.catch_warnings():
            # I don't know why this warning comes up, but at this point I don't care any more!
            warnings.filterwarnings("ignore", message="Increasing number of chunks")
            new_vars = xr.merge(
                [
                    (subset[key].diff("year") / time_diffs)
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

        new_vars.chunk(chunks).to_zarr(cache_filepath, mode="a", encoding={v: encoding for v in new_vars.data_vars})

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
    rgi_ids = pd.Series(rgi_ids.index.values, index=rgi_ids.values).drop_duplicates().astype("string")
    # rgi_ids.name = "rgi_rasterized"

    warnings.simplefilter("error")
    xr.set_options(display_max_rows=25)

    if False:
        locations = {"scheele": "RGI60-07.00283", "monaco": "RGI60-07.01494"}
        rgi_id = locations["monaco"]
        bounds = stack["bboxes"].sel(rgi_id=rgi_id).values.ravel()
        # Temporary to try things out
        stack = stack.sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[3], bounds[1]))

    stack = stack.drop_vars(["bboxes", "coordinate", "rgi_id", "rgi_index"])

    stack = stack.stack(xy=["x", "y"]).swap_dims({"xy": "rgi_rasterized"}).reset_coords(drop=True)
    stack = stack.where(stack["rgi_rasterized"] > 0, drop=True)

    if cache_path.is_file():
        os.remove(cache_path)
        # shutil.rmtree(cache_path)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        if True:
            for variable in stack.variables:
                stack[variable].attrs.clear()
                stack[variable].encoding.update(compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2))
            stack_path = temp_dir_path.joinpath("stack.zarr")
            task = stack.chunk({"rgi_rasterized": 256**2, "year": 10}).to_zarr(stack_path, compute=False)

            with TqdmCallback(desc="Saving intermediate values"):
                task.compute()

            stack = xr.open_zarr(stack_path)

        err_cols = [v for v in stack.data_vars if "_err" in v]
        count_cols = [v.replace("_err", "_count") for v in err_cols]
        scopes = [v for v in stack.data_vars if "_mask" in v]

        files = []

        with tqdm(total=len(scopes)) as progress_bar:
            for scope in scopes:
                start_time = time.time()
                filename = Path(temp_dir).joinpath(f"{scope}.zarr")
                progress_bar.desc = f"{scope}: Starting +{time.time() - start_time:.1f} s"
                progress_bar.update(0)
                filtered = stack if scope == "glacier_mask" else stack.where(stack[scope])
                progress_bar.desc = f"{scope}: Filtered +{time.time() - start_time:.1f} s"
                progress_bar.update(0)
                scoped_agg = filtered.groupby("rgi_rasterized").mean("rgi_rasterized").drop_vars(scopes)
                progress_bar.desc = f"{scope}: Grouped1 +{time.time() - start_time:.1f} s"
                progress_bar.update(0)

                pixel_counts = filtered[err_cols].notnull().groupby("rgi_rasterized").sum("rgi_rasterized")
                progress_bar.desc = f"{scope}: Grouped2 +{time.time() - start_time:.1f} s"
                progress_bar.update(0)
                scoped_agg[count_cols] = pixel_counts.rename_vars(dict(zip(err_cols, count_cols)))

                progress_bar.desc = f"{scope}: To netcdf +{time.time() - start_time:.1f} s"
                progress_bar.update(0)
                task = scoped_agg.expand_dims({"scope": np.array([scope], dtype=str)}).to_zarr(filename)
                # with TqdmCallback(desc=f"Aggregating region {attrs['label']}"):

                progress_bar.desc = f"{scope}: Done +{time.time() - start_time:.1f} s"
                files.append(filename)
                progress_bar.update()
        stack.close()

        aggs = xr.open_mfdataset(files, concat_dim="scope", combine="nested", engine="zarr")
        aggs["rgi_rasterized"] = rgi_ids.loc[aggs["rgi_rasterized"].values].values
        aggs["rgi_rasterized"] = aggs["rgi_rasterized"].astype(str)
        aggs = aggs.rename({"rgi_rasterized": "rgi_id"}).transpose("rgi_id", "year", "scope")

        attrs["creation_date"] = surgedetection.utilities.now_str()
        aggs.attrs = attrs.copy()
        # with tqdm(total=rgi_ids.shape[0], desc="Generating dask jobs") as progress_bar:
        #    agg = stack.groupby("rgi_rasterized").map(aggregate_glacier, rgi_ids=rgi_ids, progress_bar=progress_bar)
        for variable in aggs.data_vars:
            aggs[variable].encoding.update(zlib=True, complevel=5)

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        task = aggs.to_netcdf(cache_path, compute=False)
        with TqdmCallback(desc=f"Aggregating region {attrs['label']}"):
            task.compute()

        agg.close()
        scoped_agg.close()
        filtered.close()

    surgedetection.cache.symlink_to_output(cache_path, f"region_aggregates/{attrs['label']}")

    return xr.open_dataset(cache_path)
