import xarray as xr
import surgedetection.main
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
import pandas as pd


from surgedetection.constants import CONSTANTS

def main():

    stack = surgedetection.main.make_region_stack("REGN79E021X24Y05")


    locations = {
        "scheele": "RGI60-07.00283",
        "monaco": "RGI60-07.01494"
    }

    rgi_id = locations["scheele"]
    stack["glacier_mask"] = stack["rgi_rasterized"] > 0
    stack["stable_terrain_mask"] = (~stack["glacier_mask"]) & (stack["dem"] > 0.3)

    bounds = stack["bboxes"].sel(rgi_id=rgi_id).values.ravel()
    # Temporary to try things out
    stack = stack.sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[3], bounds[1]))

    stack["sar_backscatter_diff"] = stack["sar_backscatter_diff"].mean("source")
    sar_err = stack["sar_backscatter_diff"].where(stack["stable_terrain_mask"]).std(["x", "y"])
    stack["sar_backscatter_diff"] -= stack["sar_backscatter_diff"].where(stack["stable_terrain_mask"]).median(["x", "y"])

    stack["margin_mask"] = stack["glacier_mask"] & skimage.morphology.binary_dilation(~stack["glacier_mask"], footprint=np.ones((3, 3)))
    quantiles = stack["dem"].chunk(dict(x=-1, y=-1)).groupby(stack["rgi_rasterized"]).quantile([0.33, 0.67])
    stack["terminus_mask"] = stack["glacier_mask"] & (stack["dem"] <= quantiles.sel(rgi_rasterized=stack["rgi_rasterized"], quantile=0.33)).reset_coords(drop=True)
    stack["accumulation_mask"] = stack["glacier_mask"] & (stack["dem"] >= quantiles.sel(rgi_rasterized=stack["rgi_rasterized"], quantile=0.67)).reset_coords(drop=True)

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

    stack = stack.sel(time=slice("2000-01-01", None))
    err_cols = [v for v in stack.variables if "_err" in v]

    aggs = xr.Dataset()
    for i, name in enumerate(["terminus", "margin", "accumulation", "glacier"], start=1):
        agg = stack.where(stack[f"{name}_mask"]).groupby(stack["rgi_rasterized"]).mean("stacked_y_x")
        pixel_counts = stack[err_cols].notnull().groupby(stack["rgi_rasterized"]).sum("stacked_y_x")

        agg[err_cols] = agg[err_cols] / (pixel_counts.clip(min=1) ** 0.5)
        count_cols = [v.replace("_err", "_count") for v in err_cols]
        agg[count_cols] = pixel_counts.rename_vars(dict(zip(err_cols, count_cols)))

        agg = agg.drop_vars(["bboxes", "coordinate"] + [k for k in agg.variables if "_mask" in k])

        aggs = aggs.combine_first(agg.groupby(agg.time.dt.year).mean().expand_dims({"scope": [name]}))

        print(aggs)

    return
    remapping = []
    others = []
    for variable in agg.variables:
        time_coord = [c for c in agg[variable].coords if "time" in c]
        if len(time_coord) == 0:
            others.append(variable)
            continue
        remapping.append(agg[variable].rename({time_coord[0]: "time"}))

    agg = xr.merge(remapping + [agg[others]], compat="override")

    
    #agg = xr.merge([agg["dhdt"].rename({"dhdt_time": "ice_velocity_time"}), agg], compat="override")
    print(agg)
    #print(agg.rename({"ice_velocity_time": "dhdt_time"}))

    #print(pixel_counts["dhdt_err"].isel(dhdt_time=-1).clip(min=1).compute())
    #agg[err_cols] = agg[err_cols] / a

    #agg["dhdt"].isel(dhdt_time=-1).plot.hist()
    #plt.show()


    return
        

    for i, name in enumerate(["terminus", "margin", "accumulation", "glacier"], start=1):
        axis: plt.Axes = plt.subplot(2, 2, i)
        plt.title(name)

        filtered = stack.where(stack[f"{name}_mask"])

        agg = filtered.mean(["x", "y"])
        for col in agg.variables:
            if "_err" not in col:
                continue
            count = filtered[col].notnull().astype(int).sum(["x", "y"])
            agg[col] = agg[col] / (count.where(count > 0) ** 0.5)

        agg["ice_velocity2"] = agg["ice_velocity2"].where(agg["ice_velocity2_err"] < 30)

        plt.errorbar(agg["ice_velocity_time"].values, agg["ice_velocity2"], agg["ice_velocity2_err"], color="green")
        plt.ylim(-75, 75)
        plt.twinx()
        plt.errorbar(agg["dhdt_time"].values, agg["dhdt2"] * 10, agg["dhdt2_err"] * 10, color="red")
        (agg["sar_backscatter_diff"] / 2).plot(color="blue")
        plt.ylim(-1, 1)

        xlim = axis.get_xlim()
        plt.hlines(0, *xlim, color="black", linestyles="--")
        plt.xlim(np.datetime64("2009-12-01"), np.datetime64("2022-09-30"))
       

    plt.show()

