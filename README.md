
# Surge inventories
- https://zenodo.org/record/5524861

# Downloading the data
1. In `modules/earth_engine`, the Google Earth Engine scripts reside. They are separated from the rest of the repo because of all the extra dependencies that are required.
2. ERA5 data download requires a `~/.cdsapirc` file with credentials. [See the package website for further information.](https://github.com/ecmwf/cdsapi)

## Visualising the data stacks
As of December 2022, QGIS supports reading two-dimensional, but not multi-temporal (multi-dimensional) zarrs.
This means that masks, DEMs and other one-time zarrs can be visualized by just dragging and dropping the zarr into QGIS.
