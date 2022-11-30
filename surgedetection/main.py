import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features
import rasterio.warp
import shapely.wkb
import xarray as xr

import surgedetection.inputs.aster
import surgedetection.inputs


def make_glacier_stack(glims_id: str = "G014442E77835N"):

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
            bounding_box[i] -= 2 * mod
        else:
            bounding_box[i] += 2 * mod

    shape = (int((bounding_box[3] - bounding_box[1]) / mod), int((bounding_box[2] - bounding_box[0]) / mod))

    transform = rio.transform.from_bounds(*bounding_box, *(shape[::-1]))
    spatial_coords = [
        ("northing", (np.linspace(bounding_box[1] + mod / 2, bounding_box[3] - mod / 2, shape[0]))[::-1]),
        ("easting", np.linspace(bounding_box[0] + mod / 2, bounding_box[2] - mod / 2, shape[1])),
    ]
    attrs = {
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
                raster.read(1, window=window, masked=True).filled(np.nan),
                array,
                src_transform=raster.window_transform(window),
                src_crs=raster.crs,
                dst_transform=transform,
                dst_crs=crs,
                dst_resolution=mod,
                resampling=rasterio.warp.Resampling.cubic_spline,
            )

            descriptions = {
                "dhdt": "dHdt-1 averages over five-year intervals. The interval is represented by (date - 5yrs, date]",
                "dhdt_err": "dHdt-1 errors for the 'dhdt' arrays of the same date",
                "ice_velocity": "Ice surface velocities averaged per year. The exact date is in the 'ice_velocity_date' variable",
                "ice_velocity_err": "Ice surface velocity errors for the 'ice_velocity' arrays of the same date.",
                "ice_velocity_date": "Ice surface velocity dates expressed in days.",
                "sar_backscatter": "97.5th percentile-reduced backscatter over the winter. The date is the 'end date' of the interval.)"
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

    data = xr.merge(arrays)
    data.attrs.update(attrs)

    print(data)
    
    vel = data["sar_backscatter"].dropna("time", how="all").dropna("source", how="all")
    
    vel.isel(time=-1, source=0).plot()
    plt.show()
