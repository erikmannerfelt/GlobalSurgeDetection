import json
import os
import re
import time
from pathlib import Path
from typing import Any


import ee
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
import pydrive2.auth
import pydrive2.drive
import rasterio as rio
from tqdm import tqdm

YEARS = list(range(2015, 2023))

CACHE_DIR = Path("./.cache").absolute()

GDRIVE_DIR = "sentinel1"
OUTPUT_DIR = Path("../../data/sar/sentinel-1-diff/")

FILE_PATTERN = re.compile(r"^REG[N,S][0-9]{2}[W,E][0-9]{3}X[0-9]{2}Y[0-9]{2}-.*_.*\.tif$")
assert FILE_PATTERN.match("REGN76W115X01Y01-MelvilleIsland_2021-2022.tif") is not None


def get_current_rasters(output_dir: Path = OUTPUT_DIR) -> list[str]:
    labels = []
    for filepath in output_dir.glob("*.tif"):
        if FILE_PATTERN.match(filepath.name) is None:
            continue

        if "0000000000" in filepath.stem:
            labels.append(filepath.stem[:-22])
        else:
            labels.append(filepath.stem)
    
    return list(set(labels))
    return [f.stem for f in output_dir.glob("*.tif") if FILE_PATTERN.match(f.name)]


def make_all_sar_names() -> list[str]:
    regions = gpd.read_file("glacier_regions.geojson")

    labels = []

    for _, region in regions.iterrows():
        for i, year in enumerate(YEARS[1:], start=1):
            labels.append(f"{region['label']}_{YEARS[i - 1]}-{year}")

    return labels


def download_gdrive_rasters(
    gdrive_dir: str = GDRIVE_DIR, output_dir: Path = OUTPUT_DIR, progress: bool = True
) -> list[str]:

    gauth = pydrive2.auth.GoogleAuth()
    gauth.LocalWebserverAuth()  # client_secrets.json need to be in the same directory as the script
    drive = pydrive2.drive.GoogleDrive(gauth)
    root_file_list = drive.ListFile({"q": "'root' in parents and trashed=false"}).GetList()

    s1_dir_matches = [f["id"] for f in root_file_list if f["title"] == "sentinel1"]
    if len(s1_dir_matches) == 0:
        print(f"Found no {gdrive_dir} in drive")
        return []

    s1_dir_id = s1_dir_matches[0]

    s1_file_list = drive.ListFile({"q": f"'{s1_dir_id}' in parents and trashed=false"}).GetList()
    files = [f for f in s1_file_list if FILE_PATTERN.match(f["title"])]

    if len(files) == 0:
        return []

    new_rasters = []
    for gfile in tqdm(files, desc="Downloading and validating files", disable=(not progress)):
        filepath = output_dir.joinpath(gfile["title"])

        if not filepath.is_file():
            gfile.GetContentFile(filepath)

        with rio.open(filepath) as raster:
            assert raster.shape[0] > 0, f"{filepath} height is 0"
            assert raster.shape[1] > 0, f"{filepath} width is 0"
            assert "float" in raster.dtypes[0], f"{filepath} has incorrect type: {raster.dtypes[0]}"
            assert "WGS 84 / UTM zone" in raster.crs.to_wkt(), f"{filepath} has unexpected CRS: {raster.crs.to_wkt()}"

            # Try to read from the raster to make sure it works
            raster.read(1, window=rio.windows.Window(0, 0, 1, 1))

        gfile.Delete()
        new_rasters.append(filepath.stem)

    return new_rasters


def create_map() -> folium.Map:
    def add_ee_layer(self: folium.Map, ee_image_object: ee.Image, vis_params: dict[str, Any] | None, name: str) -> None:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict["tile_fetcher"].url_format,
            attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
            name=name,
            overlay=True,
            control=True,
        ).add_to(self)

    folium.Map.add_ee_layer = add_ee_layer
    map = folium.Map(location=[78, 15], zoom_start=11)

    return map


def get_start_date(year: int) -> str:
    return str(year - 1) + "-10-01"


def get_end_date(year: int) -> str:
    return str(year) + "-01-30"


def visualize(image: ee.Image, styling: dict[str, Any] | None = None) -> None:

    map = create_map()

    map.add_ee_layer(image, styling, "hello")

    map.save("index.html")


def get_scene_metadata(region: pd.Series, start_date: str, end_date: str, progress: bool = False) -> gpd.GeoDataFrame:

    full_cache_path = CACHE_DIR.joinpath(f"{region['region_id']}_{start_date}_{end_date}.feather")
    if full_cache_path.is_file():
        return gpd.read_feather(full_cache_path)

    ee.Initialize()
    sentinel1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(
        ee.Geometry.Polygon(list(region.geometry.exterior.coords))
    )

    dates = pd.date_range(start_date, end_date, freq="8W")

    filepaths = []
    params = []
    for i in range(1, len(dates)):
        start_date = dates[i - 1]
        end_date = dates[i]

        query_id = f"{region['region_id']}/{str(start_date.date())}_{str(end_date.date())}"
        cache_path = Path("./.cache").absolute().joinpath(query_id).with_suffix(".json")

        if not cache_path.is_file():
            params.append((start_date, end_date, cache_path))

        filepaths.append(cache_path)

    for (start_date, end_date, cache_path) in tqdm(
        params, disable=(len(params) == 0) or not progress, desc="Collecting metadata"
    ):

        s1_filtered = sentinel1.filterDate(ee.Date(start_date), ee.Date(end_date))

        response = s1_filtered.getInfo()["features"]

        os.makedirs(cache_path.parent, exist_ok=True)
        with open(cache_path, "w") as outfile:
            json.dump(response, outfile)

    data_list = []
    for filepath in filepaths:

        with open(filepath) as infile:
            content = infile.read()

        if len(content) == 0:
            continue
        in_data = json.loads(content)
        if len(in_data) == 0:
            continue

        data = pd.DataFrame.from_records(in_data)

        properties = pd.DataFrame.from_records(data["properties"].values)

        data_list.append(data.drop(columns=["properties"]).merge(properties, left_index=True, right_index=True))

    data = pd.concat(data_list, ignore_index=True)
    d = data["system:footprint"].to_frame("geometry")
    d[["type", "properties"]] = "Feature", None
    data = gpd.GeoDataFrame(
        data,
        geometry=gpd.GeoDataFrame.from_features(
            {"type": "FeatureCollection", "features": d.to_dict(orient="records")}, crs=4326
        ).geometry,
    ).drop(columns=["system:footprint"])

    time_columns = [
        c
        for c in data.columns
        if c in ["system:time_start", "system:time_end", "orbitProperties_ascendingNodeTime", "segmentStartTime"]
    ]
    data[time_columns] = (data[time_columns] * 1e6).astype("datetime64[ns, UTC]")

    data.to_feather(full_cache_path)

    return data


def list_running_operations(ee_initialized: bool = False) -> list[str]:
    if not ee_initialized:
        ee.Initialize()

    labels = []
    for operation in ee.data.listOperations():
        op_label = operation["metadata"]["description"]
        state = operation["metadata"]["state"]
        if state in ["PENDING", "RUNNING"]:
            labels.append(op_label)

    return labels


def process_sar_data(output_name: str, dry_run: bool = True) -> None:
    # ee.Authenticate()

    if not OUTPUT_DIR.is_dir():
        raise ValueError(f"Could not locate the current file dir: {OUTPUT_DIR}")

    region_label, year_range = output_name.split("_")
    start_year, end_year = map(int, year_range.split("-"))

    start_date = get_start_date(start_year)
    end_date = get_end_date(end_year)

    region = gpd.read_file("glacier_regions.geojson").query(f"label == '{region_label}'").iloc[0]

    resolution = int((region["xmax_proj"] - region["xmin_proj"]) / region["width_px"])

    data = get_scene_metadata(region, start_date, end_date)

    data["year"] = data["system:time_start"].dt.year
    data.loc[data["system:time_start"].dt.month >= 10, "year"] += 1
    data.sort_values("year", inplace=True)

    data["polarisations-str"] = data["transmitterReceiverPolarisation"].astype("string")

    years = np.sort(data["year"].unique())
    labels = [f"{region['label']}_{year - 1}-{year}" for year in years[1:]]

    ee.Initialize()

    sentinel1 = ee.ImageCollection("COPERNICUS/S1_GRD")
    max_images = pd.Series(
        dtype="object", index=pd.MultiIndex.from_arrays([[]] * 4, names=["year", "kind", "orbit_nr", "pol"])
    )
    for (year, mode, orbit, orbit_nr), images in data.groupby(
        ["year", "instrumentMode", "orbitProperties_pass", "relativeOrbitNumber_start"]
    ):
        for pol in np.unique(images["transmitterReceiverPolarisation"].apply(list))[0]:
            images_with_pol = images[images["polarisations-str"].str.contains(pol)]
            max_images.loc[(year, mode + "-" + orbit, orbit_nr, pol)] = (
                sentinel1.filter(ee.Filter.inList("system:index", images_with_pol["system:index"].values.tolist()))
                .select(pol)
                .max()
            )

    diffs = pd.DataFrame(dtype="object", columns=["bands", "image"])
    for (kind, pol), images0 in max_images.groupby(level=[1, 3]):

        years = np.sort(np.unique(images0.index.get_level_values(0)))
        if len(years) < 2:
            continue
        for i in range(1, len(years)):
            year_before = years[i - 1]
            year_after = years[i]
            interval = f"{year_before}-{year_after}"

            if f"{region['label']}_{interval}" not in labels:
                continue

            year_diffs = []
            for (_, orbit_nr, _), image_after in images0.loc[pd.IndexSlice[year_after, :, :, :]].items():

                index_before = (year_before, kind, orbit_nr, pol)

                if not images0.index.isin([index_before]).any():
                    continue

                image_before = images0.loc[index_before]

                year_diffs.append(image_after.subtract(image_before))

            band_name = f"{kind}-{pol}"
            image = ee.ImageCollection(year_diffs).mean().rename(band_name)
            interval = f"{year_before}-{year_after}"

            if diffs.index.isin([interval]).any():
                diffs.loc[interval, "image"] = diffs.loc[interval, "image"].addBands(image)
                diffs.loc[interval, "bands"] += "/" + band_name
            else:
                diffs.loc[interval, ["image", "bands"]] = image, band_name

    for interval, diff in diffs["image"].items():
        label = f"{region['label']}_{interval}"
        job = ee.batch.Export.image.toDrive(
            image=diff,
            description=label,
            folder=GDRIVE_DIR,
            dimensions=f"{region['width_px']}x{region['height_px']}",
            crs=f"epsg:{region['crs_epsg']}",
            maxPixels=1e13,
            crsTransform=[resolution, 0, region["xmin_proj"], 0, -resolution, region["ymax_proj"]],
        )
        if dry_run:
            print(f"Not submitting job {dry_run=}")
        else:
            job.start()


def diff_list(first: list[str], second: list[str]) -> list[str]:
    return [s for s in first if s not in second]


def get_glacier_diffs(dry_run: bool = False):
    glacier_name = "Strongbreen"
    bounds = {"left": 17.09, "bottom": 77.52, "right": 18.06, "top": 77.74}
    resolution = 50

    bbox = shapely.geometry.box(bounds["left"], bounds["bottom"], bounds["right"], bounds["top"])

    bounds_proj = gpd.GeoSeries(bbox, crs=4326).to_crs(32633).total_bounds

    for i in range(bounds_proj.shape[0]):
        bounds_proj[i] -= bounds_proj[i] % resolution
        if i > 1:
            bounds_proj[i] += resolution
            

    width = int((bounds_proj[2] - bounds_proj[0]) / resolution)
    height = int((bounds_proj[3] - bounds_proj[1]) / resolution)

    region = gpd.read_file("glacier_regions.geojson").query("name == 'Svalbard'").iloc[0]

    metadata = get_scene_metadata(region, "2014-10-01", "2022-09-30", progress=True) 
    metadata["geometry"] = metadata["geometry"].apply(shapely.geometry.Polygon)
    metadata = metadata[metadata.contains(bbox)]

    #metadata = pd.concat([metadata[metadata["resolution"] == "H"], metadata[metadata["resolution"] != "H"]]).drop_duplicates(subset=[")
    metadata = metadata.sort_values("resolution_meters").drop_duplicates(subset=["segmentStartTime", "orbitNumber_start"]).sort_values("system:time_start")
    metadata["polarisation"] = metadata["transmitterReceiverPolarisation"].apply(lambda l: "/".join(l))
    #import matplotlib.pyplot as plt
    #metadata.plot()
    #plt.show()

    diffs_i = []
    diffs_d = []
    for (orbit_nr, orbit, mode), meta0 in metadata.groupby(["relativeOrbitNumber_start", "orbitProperties_pass", "instrumentMode"], as_index=False):
        for pol in ["HH", "HV", "VH", "VV"]:
            meta1 = meta0[meta0["polarisation"].str.contains(pol)]

            if meta1.shape[0] < 2:
                continue


            for i in range(1, meta1.shape[0]):
                scene0 = meta1.iloc[i - 1]
                scene1 = meta1.iloc[i]

                if (scene1["system:time_start"] - scene0["system:time_start"]) > pd.Timedelta(days=50):
                    continue

                diffs_i.append(
                    (mode,orbit,pol, pd.Interval(pd.Timestamp(scene0["system:time_start"]), pd.Timestamp(scene1["system:time_start"])))
                )
                diffs_d.append({
                    "id_0": scene0["id"],
                    "id_1": scene1["id"],
                })


    diffs = pd.DataFrame(data=diffs_d, index=pd.MultiIndex.from_tuples(diffs_i, names=["mode", "orbit", "pol","interval"]))

    ee.Initialize()
    sentinel1 = ee.ImageCollection("COPERNICUS/S1_GRD")


    for (mode, year), year_diffs in diffs.groupby([diffs.index.get_level_values(0), diffs.index.get_level_values(3).mid.year]):

        #if (mode == "IW" and year == 2015):
        #    continue

        out = None
        for (_, orbit, pol, interval), scenes in year_diffs.iterrows():

            name = f"{mode}_{orbit}_{pol}_{interval.left.isoformat()}_{interval.right.isoformat()}"

            length_days = interval.length.total_seconds() / (3600 * 24)

            scene0 = sentinel1.filter(ee.Filter.stringContains("system:index", scenes["id_0"])).select([pol]).first()
            scene1 = sentinel1.filter(ee.Filter.stringContains("system:index", scenes["id_1"])).select([pol]).first()

            diff = scene1.subtract(scene0).divide(length_days).rename([name])

            if out is None:
                out = diff
            else:
                out = out.addBands(diff)

        label = f"{glacier_name}-diffs-{mode}-{year}"
        job = ee.batch.Export.image.toDrive(
            image=out,
            description=label,
            folder=GDRIVE_DIR,
            dimensions=f"{width}x{height}",
            crs=f"epsg:{region['crs_epsg']}",
            maxPixels=1e13,
            crsTransform=[resolution, 0, bounds_proj[0], 0, -resolution, bounds_proj[3]],
        )
        print(mode, year, year_diffs.shape[0])
        #str(out)
        if dry_run:
            print(f"Not submitting job {dry_run=}")
        else:
            job.start()
    



def main(max_concurrent_processes: int = 10) -> None:

    all_sar_names = make_all_sar_names()

    downloaded_names = get_current_rasters()

    print(f"Found {len(downloaded_names)} (out of {len(all_sar_names)}) downloaded files")

    non_downloaded_names = diff_list(all_sar_names, downloaded_names)
    non_downloaded_names.sort()

    if len(non_downloaded_names) == 0:
        print("All rasters downloaded")
        return

    ee.Initialize()

    with tqdm(total=len(non_downloaded_names), desc="Submitting processes", smoothing=0) as progress_bar:
        while len(non_downloaded_names) > 0:
            new_downloads = download_gdrive_rasters(progress=False)
            if len(new_downloads) > 0:
                non_downloaded_names = diff_list(non_downloaded_names, new_downloads)
                progress_bar.set_description(f"Downloaded {len(new_downloads)} new file(s)")
                progress_bar.update(len(new_downloads))

            names_in_progress = list_running_operations(ee_initialized=True)

            unstarted_names = diff_list(non_downloaded_names, names_in_progress)
            if len(unstarted_names) == 0:
                return

            if len(names_in_progress) < max_concurrent_processes:
                process_sar_data(unstarted_names[0], dry_run=False)
                progress_bar.set_description(f"Queried {unstarted_names[0]}")

            else:
                progress_bar.set_description(f"Awaiting {len(names_in_progress)} processes.")
                time.sleep(2 * 60)


if __name__ == "__main__":
    main()
