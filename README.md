
# Downloading the data
1. In `modules/earth_engine`, the Google Earth Engine scripts reside. They are separated from the rest of the repo because of all the extra dependencies that are required.
2. The RGI6 dataset needs downloading from the [NSIDC data portal](https://nsidc.org/data/nsidc-0770/versions/6). [Here is a direct link to the complete dataset](https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v6/nsidc0770_00.rgi60.complete.zip).
3. Alternatively, the RGI6 dataset can be downloaded here: http://www.glims.org/RGI/rgi60_files/00_rgi60.zip

## Visualising the data stacks
As of December 2022, QGIS supports reading two-dimensional, but not multi-temporal (multi-dimensional) zarrs.
This means that masks, DEMs and other one-time zarrs can be visualized by just dragging and dropping the zarr into QGIS.
