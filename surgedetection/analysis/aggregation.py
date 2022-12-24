import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
import pandas as pd
import dask
from tqdm import tqdm
from tqdm.dask import TqdmCallback
import warnings
import tempfile
from pathlib import Path
import concurrent.futures
import time
import os
import shutil
from dask.diagnostics import ProgressBar
import threading
import zarr

NS_IN_A_YEAR=np.timedelta64(1, "Y").astype("timedelta64[ns]").astype(float)

import surgedetection.main
import surgedetection.cache
import surgedetection.utilities
from surgedetection.constants import CONSTANTS

def subset_glacier(stack: xr.Dataset, rgi_id: str) -> xr.Dataset:
    bounds = stack["bboxes"].sel(rgi_id=rgi_id).values.ravel()
    subset = stack.sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[3], bounds[1]))
    subset.attrs["rgi_id"] = rgi_id
    return subset


def agg_glacier(subset: xr.Dataset) -> xr.DataArray:
    rgi_id = subset.attrs["rgi_id"]
    subset["glacier_mask"] = subset["rgi_rasterized"] == subset["rgi_index"].sel(rgi_id=rgi_id)
    subset = subset.drop_vars(["bboxes", "rgi_index", "rgi_rasterized", "rgi_id", "coordinate"])

    quantiles = subset["dem"].where(subset["glacier_mask"]).chunk(dict(x=-1, y=-1)).quantile([0.33, 0.67])
    subset["terminus_mask"] = subset["glacier_mask"] & (subset["dem"] <= quantiles.sel(quantile=0.33)).reset_coords(drop=True)
    subset["accumulation_mask"] = subset["glacier_mask"] & (subset["dem"] >= quantiles.sel(quantile=0.67)).reset_coords(drop=True)
    subset["margin_mask"] = subset["glacier_mask"] & skimage.morphology.binary_dilation(~subset["glacier_mask"], footprint=np.ones((3, 3)))

    for key in ["dhdt", "ice_velocity"]:
        new_col = f"{key}2"
        new_err = f"{new_col}_err"

        times = np.array(subset[key].attrs["times"], dtype="datetime64[ns]")
        time_diffs = xr.DataArray(np.diff(times).astype(float) / NS_IN_A_YEAR, coords={"time": times[1:]})
        time_filtered = subset[[key, f"{key}_err"]].dropna("time", how="all")
        subset[new_col] = time_filtered[key].diff("time").chunk({"time": 1}) / time_diffs
        subset[new_err] = ((time_filtered[f"{key}_err"].isel(time=slice(1, None)) ** 2) + (time_filtered[f"{key}_err"].isel(time=slice(-1)).values ** 2) ** 0.5) / time_diffs

        #time_diffs = time_filtered["time"].diff("time").astype(float) / (np.timedelta64(1, "Y").astype("timedelta64[ns]").astype(float))
        #for col in [new_col, new_err]:
        #    subset[col] = subset[col] / time_diffs

    # The dhdt2 and *_err make massive chunks for some reason
    subset = subset.chunk({"time": 1})
    masks = [name for name in subset.variables if "_mask" in name]
    err_vars = [name for name in subset.variables if "_err" in name]
    count_vars = [v.replace("_err", "_count") for v in err_vars]

    aggs = []
    for mask_name in masks:
        agg = subset.drop_vars(masks).where(subset[mask_name]).mean(["x", "y"])
        pixel_counts = subset[err_vars].where(subset[mask_name]).notnull().sum(["x", "y"])
        agg[err_vars] = agg[err_vars] / (pixel_counts.clip(min=1) ** 0.5)
        agg[count_vars] = pixel_counts.rename(dict(zip(err_vars, count_vars)))

        aggs.append(agg.groupby(agg.time.dt.year).mean().expand_dims({"scope": [mask_name.replace("_mask", "")]}, axis=1))

    return xr.concat(aggs, "scope").expand_dims({"rgi_id": [rgi_id]})


def agg_naive(stack):
    out_path = Path("hello.zarr")

    if out_path.is_dir():
        shutil.rmtree(out_path)
    start_time = time.time()

    aggs = []
    first = True
    with tqdm(total=stack["rgi_id"].shape[0]) as progress_bar:
        for i, rgi_id in enumerate(stack["rgi_id"].values):
            progress_bar.desc = rgi_id

            aggs.append(agg_glacier(subset_glacier(stack, rgi_id)))

            if i % 2 == 0:
                args = {} if first else {"mode": "a", "append_dim": "rgi_id"}
                xr.concat(aggs, "rgi_id").to_zarr(out_path, **args)
                first = False


                aggs.clear()

            progress_bar.update()
    

def interpret_stack(region_id: str = "REGN79E021X24Y05", force_redo: bool = False) -> xr.Dataset:

    cache_filepath = surgedetection.cache.get_cache_name(f"interpret_stack-{region_id}", extension=".zarr")
    if (not force_redo) and cache_filepath.is_dir():
        return xr.open_zarr(cache_filepath)

    stack = surgedetection.main.make_region_stack(region_id)

    attrs = stack.attrs.copy()
    if "creation_date" in attrs:
        attrs["source_creation_date"] = attrs["creation_date"]

    attrs["creation_date"] = surgedetection.utilities.now_str()
    var_attrs = {v: stack[v].attrs.copy() for v in stack.data_vars}

    if False:
        locations = {
            "scheele": "RGI60-07.00283",
            "monaco": "RGI60-07.01494"
        }
        rgi_id = locations["monaco"]
        bounds = stack["bboxes"].sel(rgi_id=rgi_id).values.ravel()
         #Temporary to try things out
        stack = stack.sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[3], bounds[1]))

    vars_with_time = [v for v in stack.variables if "time" in stack[v].dims and v != "time"]
    stack[vars_with_time] = stack[vars_with_time].groupby("time.year").mean().squeeze()

    stack["sar_backscatter_diff"] = stack["sar_backscatter_diff"].mean("source")
    stack["sar_backscatter"] = stack["sar_backscatter"].mean("source")
    stack = stack.drop_dims(["source", "time"])
    
    stack["glacier_mask"] = stack["rgi_rasterized"] > 0
    stack["margin_mask"] = (stack["glacier_mask"] & skimage.morphology.binary_dilation(~stack["glacier_mask"], footprint=np.ones((3, 3)))).compute()

    stack["dem"].load()
    stack["rgi_rasterized"].load()
    for_quantiles = stack[["dem", "rgi_rasterized"]].stack(xy={"x", "y"}).swap_dims(xy="rgi_rasterized")
    quantiles = for_quantiles["dem"].groupby("rgi_rasterized").quantile([0.33, 0.67]).sel(rgi_rasterized=stack["rgi_rasterized"])

    stack["terminus_mask"] = stack["glacier_mask"] & (stack["dem"] <= quantiles.sel(quantile=0.33)).reset_coords(drop=True) 
    stack["accumulation_mask"] = stack["glacier_mask"] & (stack["dem"] >= quantiles.sel(quantile=0.67)).reset_coords(drop=True) 

    chunks = {"x": 256, "y": 256, "year": 1}
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    save_params = {v: {"compressor": compressor} for v in stack.variables}

    if cache_filepath.is_dir():
        shutil.rmtree(cache_filepath)

    stack.attrs = attrs
    for variable in var_attrs:
        stack[variable].attrs = var_attrs[variable]

    task = stack.chunk(chunks).to_zarr(cache_filepath, compute=False, encoding=save_params)

    with TqdmCallback(desc=f"A: Computing region {region_id}"):
        task.compute()

    stack = xr.open_zarr(cache_filepath)

    for key in tqdm(["ice_velocity", "dhdt"], desc="B: Calculating acceleration variables"):
        new_var_name = f"{key}2"
        new_err_name = f"{new_var_name}_err"

        #times = np.array(stack[key].attrs["times"], dtype="datetime64[ns]")
        subset = stack[[key, f"{key}_err"]].dropna("year", "all")

        #subset = stack[[key, f"{key}_err"]].sel(time=times)
        time_diffs = subset["year"].diff("year")

        new_vars = xr.merge([
            (subset[key].diff("year").chunk(year=30) / time_diffs).reindex({"year": stack["year"]}).rename(new_var_name).astype(stack[key].dtype),
            (((subset[f"{key}_err"].isel(year=slice(1, None)) ** 2) + (subset[f"{key}_err"].isel(year=slice(-1)).values ** 2) ** 0.5) / time_diffs).reindex({"year": stack["year"]}).rename(new_err_name).astype(stack[f"{key}_err"].dtype)
        ])
        new_vars.attrs = stack.attrs.copy()

        for old_key, new_key in [(key, new_var_name), (f"{key}_err", new_err_name)]:
            new_vars[new_key].attrs = subset[old_key].attrs.copy()

        save_params = {v: {"compressor": compressor} for v in [new_var_name, new_err_name]}
        new_vars.chunk(chunks).to_zarr(cache_filepath, mode="a", encoding=save_params)


    surgedetection.cache.symlink_to_output(cache_filepath, f"region_stacks/{attrs['label']}")

    return xr.open_zarr(cache_filepath)
    



def main(region_id: str = "REGN79E021X24Y05", force_redo: bool = False):

    cache_path = surgedetection.cache.get_cache_name(f"aggregate_region-{region_id}", extension=".nc")

    if (not force_redo) and cache_path.is_file():
        return xr.open_dataset(cache_path)
        #return xr.open_zarr(cache_path)
    stack = interpret_stack(region_id=region_id)

    attrs = stack.attrs.copy()
    if "creation_date" in attrs:
        attrs["source_creation_date"] = attrs["creation_date"]


    rgi_ids = stack["rgi_index"].to_series().sort_index()
    rgi_ids = pd.Series(rgi_ids.index.values, index=rgi_ids.values).astype("string")
    #rgi_ids.name = "rgi_rasterized"

    warnings.simplefilter("error")
    xr.set_options(display_max_rows=25)

    if False:
        locations = {
            "scheele": "RGI60-07.00283",
            "monaco": "RGI60-07.01494"
        }
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
        #shutil.rmtree(cache_path)

    stack_size = 3
    n = np.unique(stack["rgi_rasterized"].values).shape[0]
    n_stacks = int(np.ceil(n / stack_size))
    first = True
    #compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    aggs = []
    with tqdm(total=n_stacks) as progress_bar, tempfile.TemporaryDirectory() as temp_dir:

        temp_filename = Path(temp_dir).joinpath("arr.zarr")
        for i, (rgi_index, per_rgi) in enumerate(stack.groupby("rgi_rasterized")):
            per_scope = []
            #per_rgi = per_rgi.compute()  # For some reason, this makes it all faster
            for scope in scopes:
                filtered = per_rgi.where(per_rgi[scope])
                scoped_agg = filtered.mean("rgi_rasterized").drop_vars(scopes)
                pixel_counts = filtered[err_cols].notnull().sum("rgi_rasterized")
                scoped_agg[count_cols] = pixel_counts.rename_vars(dict(zip(err_cols, count_cols)))

                per_scope.append(scoped_agg.expand_dims({"scope": np.array([scope], dtype=str)}))

            aggs.append(xr.concat(per_scope, "scope").expand_dims({"rgi_id": np.array([rgi_ids[rgi_index]], dtype=str)}))
            if len(aggs) >= stack_size or i == (n - 1): 
                #save_params = {v: {"compressor": compressor} for v in aggs[0].data_vars}
                if first:
                    zarr_args = {} # {"encoding": save_params} 
                else:
                      zarr_args = {"mode": "a", "append_dim": "rgi_id"}
                xr.concat(aggs, "rgi_id").to_zarr(temp_filename, **zarr_args)
                progress_bar.update()
                aggs.clear()
                if not first:
                    break
                first = False

        output = xr.open_zarr(temp_filename)

        for variable in output.data_vars:
            output[variable].encoding.update(zlib=True, complevel=9)

        attrs["creation_date"] = surgedetection.utilities.now_str()
        output.attrs = attrs.copy()

        output.to_netcdf(cache_path)

    surgedetection.cache.symlink_to_output(cache_path, f"region_aggregates/{attrs['label']}")
    


    
    return xr.open_dataset(cache_path)
    #rgi_iter = [(rgi_ids[i], arr) for i, arr in ]

    with tqdm(total=np.unique(stack["rgi_rasterized"].values).shape[0], desc="Aggregating each glacier") as progress_bar:

        def agg(per_rgi: tuple[str, xr.Dataset], zarr_args: dict[str, str] = {"mode": "a", "append_dim": "rgi_id"}) -> xr.Dataset:
            rgi_id, per_rgi = per_rgi
            per_scope = []
            for scope in scopes.copy():

                filtered = per_rgi.where(per_rgi[scope])
                scoped_agg = filtered.mean("rgi_rasterized").drop_vars(scopes)
                pixel_counts = filtered[err_cols].notnull().sum("rgi_rasterized")
                scoped_agg[count_cols] = pixel_counts.rename_vars(dict(zip(err_cols, count_cols)))

                scoped_agg = (scoped_agg.groupby(scoped_agg.time.dt.year).mean().expand_dims({"scope": [scope]}))

                per_scope.append(scoped_agg)

            aggs = xr.concat(per_scope, "scope").expand_dims({"rgi_id": [rgi_id]})
            aggs["rgi_id"] = aggs["rgi_id"].astype(str)
            aggs["scope"] = aggs["scope"].astype(str)
            aggs.to_zarr(output_path, **zarr_args)
            progress_bar.update()
        
        agg(rgi_iter.pop(0),zarr_args={"mode": "w"})
        

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(agg, rgi_iter)) 
        #for per_rgi in rgi_iter:
        #    agg(per_rgi, progress_bar=progress_bar, output_path=output_path, rgi_ids=rgi_ids)
        #aggs = .map(agg, progress_bar=progress_bar, output_path=output_path, rgi_ids=rgi_ids)
    task = aggs.to_netcdf("hello.nc", compute=False)
    with TqdmCallback(desc="Saving result"):
        task.compute()
    return

    aggs = []
    scopes = ["terminus", "margin", "accumulation", "glacier"]
    #with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    print("Making aggregates")
    for name in scopes:
        print(name)
        filtered = stack.where(stack[f"{name}_mask"])
        print(filtered)
        agg = filtered.groupby("rgi_rasterized").mean("rgi_rasterized")
        print("Done with main agg")

        pixel_counts = stack[err_cols].where(stack[f"{name}_mask"]).notnull().groupby("rgi_rasterized").sum("rgi_rasterized")

        agg[err_cols] = agg[err_cols] / (pixel_counts.clip(min=1) ** 0.5)
        agg[count_cols] = pixel_counts.rename_vars(dict(zip(err_cols, count_cols)))

        #agg["rgi_id"] = agg["rgi_id"].isel(rgi_id=agg["rgi_rasterized"].astype(int))
        agg = (agg.drop_vars([k for k in agg.variables if "_mask" in k]).groupby(agg.time.dt.year).mean().expand_dims({"scope": [name]}))
        #agg["rgi_index"] = agg["rgi_index"].isel(rgi_index=agg["rgi_rasterized"])

        aggs.append(agg)


    aggs = xr.concat(aggs, "scope")
    aggs = aggs.assign_coords({"rgi_rasterized": rgi_ids.sel(rgi_rasterized=aggs["rgi_rasterized"])}).rename({"rgi_rasterized": "rgi_id"})

    print(aggs)

    task = aggs.to_netcdf("hello.nc", compute=False)

    with TqdmCallback():
        task.compute()
    print(aggs)

    
    return

    #subsets = [(subset_glacier(stack, rgi_id), rgi_id) for rgi_id in stack["rgi_id"].values]

    #aggs = []
    #for args in tqdm(subsets[:5]):
    #    aggs.append(agg_glacier(*args))
        

    #print("hello")
    #agg_glacier(stack, rgi_id)
    #return

    stack["stable_terrain_mask"] = (~stack["glacier_mask"]) & (stack["dem"] > 0.3)

    #sar_err = stack["sar_backscatter_diff"].where(stack["stable_terrain_mask"]).std(["x", "y"])
    #stack["sar_backscatter_diff"] -= stack["sar_backscatter_diff"].where(stack["stable_terrain_mask"]).median(["x", "y"])

    non_mask_vars = [var for var in stack.variables if "_mask" not in var and "x" in stack[var].dims and var != "x"]
    stack[non_mask_vars] = stack[non_mask_vars].where(stack["glacier_mask"])


    print("Computing margin mask")

    #stack = stack.where(stack["glacier_mask"])
    print("Computing DEM quantiles")

    stacking = {"xy": ["x", "y"]}
    for_quantiles = stack[["dem", "rgi_rasterized"]].where(stack["glacier_mask"]).stack(stacking).dropna("xy", subset=["rgi_rasterized"]).chunk(dict(xy=-1))
    # The sel part takes forever! Better ways of doing it perhaps?
    quantiles = for_quantiles["dem"].groupby(for_quantiles["rgi_rasterized"].astype(int)).quantile([0.33, 0.67]).sel(rgi_rasterized=stack["rgi_rasterized"], method="pad").reset_coords(drop=True)
    del for_quantiles

    print("Calculating terminus mask")
    stack["terminus_mask"] = stack["glacier_mask"] & (stack["dem"] <= quantiles.sel(quantile=0.33)).fillna(False)

    print("Calculating accumulation mask")
    stack["accumulation_mask"] = stack["glacier_mask"] & (stack["dem"] <= quantiles.sel(quantile=0.67)).fillna(False)


    #stack = stack.where(stack["rgi_mask"] == loc["id"])
    for key in ["dhdt", "ice_velocity"]:
        new_col = f"{key}2"
        new_err = f"{new_col}_err"

        times = np.array(stack[key].attrs["times"], dtype="datetime64[ns]")
        subset = stack[[key, f"{key}_err"]].sel(time=times)
        stack[new_col] = subset[key].diff("time")
        stack[new_err] = ((subset[f"{key}_err"].isel(time=slice(1, None)) ** 2) + (subset[f"{key}_err"].isel(time=slice(-1)).values ** 2) ** 0.5)

        time_diffs = subset["time"].diff("time").astype(float) / (np.timedelta64(1, "Y").astype("timedelta64[ns]").astype(float))
        for col in [new_col, new_err]:
            stack[col] = stack[col] / time_diffs

    # The dhdt2 and *_err made massive chunks for some reason
    #stack = stack.chunk({"time": 1})

    stack = stack.sel(time=slice("2000-01-01", None))
    err_cols = [v for v in stack.variables if "_err" in v]

    aggs = []
    scopes = ["terminus", "margin", "accumulation", "glacier"]
    #with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    warnings.simplefilter("error")
    print("Making aggregates")
    if True:
        for i, name in enumerate(scopes, start=1):
            agg = stack.where(stack[f"{name}_mask"]).groupby(stack["rgi_rasterized"]).mean("stacked_y_x")

            pixel_counts = stack[err_cols].where(stack[f"{name}_mask"]).notnull().groupby(stack["rgi_rasterized"]).sum("stacked_y_x")

            agg[err_cols] = agg[err_cols] / (pixel_counts.clip(min=1) ** 0.5)
            count_cols = [v.replace("_err", "_count") for v in err_cols]
            agg[count_cols] = pixel_counts.rename_vars(dict(zip(err_cols, count_cols)))

            agg["rgi_id"] = agg["rgi_id"].isel(rgi_id=agg["rgi_rasterized"].astype(int))
            agg = (agg.drop_vars(["bboxes", "coordinate", "rgi_index"] + [k for k in agg.variables if "_mask" in k]).groupby(agg.time.dt.year).mean().expand_dims({"scope": [name]}).swap_dims({"rgi_rasterized": "rgi_id"}).drop_vars("rgi_rasterized"))
            #agg["rgi_index"] = agg["rgi_index"].isel(rgi_index=agg["rgi_rasterized"])

            aggs.append(agg)


    aggs = xr.concat(aggs, "scope")
    #aggs.to_dataframe(["rgi_id", "year", "scope"]).to_csv("hello.csv")
    aggs.to_netcdf("hello.nc")

