import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import shapely.geometry
from tqdm import tqdm

import surgedetection.inputs.rgi
from surgedetection.constants import CONSTANTS

UTM_ZONE_LON_WIDTH = 6
MAX_ZONE_LON_WIDTH = 8
MAX_PIXEL_COUNT = 100000


def get_best_utm_zone(longitude: float) -> int:
    return max(min(round((longitude + 180) / UTM_ZONE_LON_WIDTH) + 1, 60), 1)


def parse_manual_glacier_zones(filename: str = "Glacier_zones.ods") -> pd.DataFrame:
    filepath = CONSTANTS.manual_input_data_path.joinpath(filename)
    entries = pd.read_excel(filepath).dropna(how="all")

    output = []
    for _, row in entries.iterrows():

        rgi_zones = row["RGI"].split("/")

        cut_geom = shapely.geometry.box(
            minx=row["CutWest"] if np.isfinite(row["CutWest"]) else -179.99999,
            miny=row["CutSouth"] if np.isfinite(row["CutSouth"]) else -90,
            maxx=row["CutEast"] if np.isfinite(row["CutEast"]) else 180,
            maxy=row["CutNorth"] if np.isfinite(row["CutNorth"]) else 90,
        )

        output.append(
            {
                "name": row["Name"],
                "rgi_regions": rgi_zones,
                "O1_regions": list({int(zone.split("-")[0]) for zone in rgi_zones}),
                "O2_regions": list({int(zone.split("-")[1]) if "-" in zone else 1 for zone in rgi_zones}),
                "crs": pyproj.CRS.from_string(f"WGS 84 / UTM Zone {int(row['UTM zone'])}{row['North/South']}"),
                "cut_geom": cut_geom,
            }
        )
    return pd.DataFrame.from_records(output)


def geoseries_from_bounds(xmin: float, ymin: float, xmax: float, ymax: float, crs: pyproj.CRS) -> gpd.GeoSeries:
    return gpd.GeoSeries([shapely.geometry.box(xmin, ymin, xmax, ymax)], crs=crs)


def generate_safe_wgs_bbox(bounds: list[float], crs: pyproj.CRS, resolution: int = 100) -> gpd.GeoSeries:
    """
    Generate a bounding box in WGS84 given the bounds in a specified CRS.

    It's hard because a simple projection with a 4-vertex rectangle is accurate in the
    exact edges, but not in between, so the rectangle has to be interpolated.

    Arguments
    ---------
    bounds: A list of [xmin, ymin, xmax, ymax] bounding coordinates
    crs: The CRS of the bounding coordinates
    resolution: The approximate vertex count of the outgoing bounding rectangle.

    Returns
    -------
    A rectangle showing the projected bounds of the provided bounding coordinates.
    """
    # Specify the four edges of the bounding box
    # box_edges = np.array([
    #    [bounds[2], bounds[1]],  # lr
    #    [bounds[2], bounds[3]],  # ur
    #    [bounds[0], bounds[3]],  # ul
    #    [bounds[0], bounds[1]],  # ll
    # ])
    box = shapely.geometry.box(*bounds).exterior

    # The vertex interpolation works relatively, so the first part makes 97 vertices (the start plus 96 in between)
    # The three other edges are found through this convolued relative cumulative distance call
    relative_vertex_points = np.unique(
        np.r_[
            np.linspace(0, 1 - (1 / resolution), resolution - 3),
            np.cumsum(np.sqrt(np.sum(np.square(np.diff(box.xy, axis=1)), axis=0)))[:-1] / box.length,
        ]
    )

    polygon = shapely.geometry.Polygon(
        shapely.geometry.LinearRing([box.interpolate(d, normalized=True) for d in relative_vertex_points])
    )

    return gpd.GeoSeries([polygon], crs=crs).to_crs(4326)

    return
    xmin, ymin, xmax, ymax = bounds
    target_crs = 4326

    xmean = np.mean([xmin, xmax])
    ymean = np.mean([ymin, ymax])

    ymean, xmean = np.ravel(pyproj.Transformer.from_crs(crs, target_crs).transform([xmean], [ymean]))

    data = gpd.GeoDataFrame(
        [
            ["ll", xmin, ymin],
            ["ul", xmin, ymax],
            ["ur", xmax, ymax],
            ["lr", xmax, ymin],
        ],
        columns=["name", "x", "y"],
    )
    data["geometry"] = gpd.points_from_xy(data["x"], data["y"], crs=crs).to_crs(4326)

    for i, row in data.iterrows():

        lon_diff = xmean - row.geometry.x
        if abs(lon_diff) > 90:
            data.loc[i, "geometry"] = shapely.geometry.Point(180 if lon_diff >= 0 else -180, row.geometry.y)

    return gpd.GeoSeries(
        [shapely.geometry.Polygon(np.r_[data["geometry"].values, data["geometry"].values[0:1]].tolist())],
        crs=target_crs,
    )


def make_glacier_zones(buffer: float = 5000.0, mod: float = 1000.0, area_threshold: float = 1.6) -> gpd.GeoDataFrame:
    manual_entries = parse_manual_glacier_zones()  # .query("name == 'NE Russia'")

    rgi_o2_regions = surgedetection.inputs.rgi.read_rgi6_regions()

    cache_path = surgedetection.cache.get_cache_name(
        "make_glacier_zones", [buffer, mod, area_threshold], extension="geojson"
    )

    # All glaciers with fewer than 4x4 pixels are excluded (400 * 400 m == 1.6 km²)
    rgi = surgedetection.inputs.rgi.read_all_rgi6(query=f"Area > {area_threshold}")

    excluded_antarctica_zones = [str(v) for v in (list(range(11, 19)) + list(range(22, 25)))]

    rgi.drop(index=rgi[(rgi["O1Region"] == "19") & rgi["O2Region"].isin(excluded_antarctica_zones)].index, inplace=True)
    # rgi.drop(rgi[((rgi["O1Region"] == "19") & rgi["O2Region"].isin([excluded_antarctica_zones])), inplace=True)
    rgi["O1Region"] = rgi["O1Region"].astype(int)

    zones_list = []
    # rgi_outlines = {}
    for _, entry in tqdm(
        manual_entries.iterrows(), total=manual_entries.shape[0], desc="Calculating optimal glacier zones"
    ):

        rgi_region_wgs84 = rgi_o2_regions[rgi_o2_regions["RGI_CODE"].isin(entry["rgi_regions"])].dissolve()
        region_polygons_wgs84 = rgi_region_wgs84.intersection(entry["cut_geom"])
        if region_polygons_wgs84.shape[0] == 0:
            raise ValueError(f"Error on region {entry}. RGI region doesn't overlap")
        region_polygon_wgs84 = region_polygons_wgs84.values[0]

        outlines = rgi[rgi["O1Region"].isin(entry["O1_regions"])]
        outlines = outlines[outlines.intersects(region_polygon_wgs84)].to_crs(entry["crs"])

        if outlines.shape[0] == 0:
            continue

        bounds = []
        for i, bound in enumerate(outlines.total_bounds):
            bound -= bound % mod
            bound -= buffer * (1 if i < 2 else -1)
            bounds.append(bound)
        width = int((bounds[3] - bounds[1]) / CONSTANTS.pixel_size)
        height = int((bounds[2] - bounds[0]) / CONSTANTS.pixel_size)

        # bounding_box_wgs84 = gpd.GeoSeries([shapely.geometry.box(*bounds).segmentize(max_segment_length=10000)], crs=entry["crs"]).to_crs(4326)

        bounding_box_wgs84 = generate_safe_wgs_bbox(bounds, outlines.crs)

        zones_list.append(
            {
                "name": entry["name"],
                "O1_regions": "/".join(map(str, entry["O1_regions"])),
                "O2_regions": "/".join(map(str, entry["O2_regions"])),
                "crs": entry["crs"].to_wkt(),
                "n_glaciers": outlines.shape[0],
                "glacier_area": outlines["Area"].sum(),
                "xmin_proj": bounds[0],
                "ymin_proj": bounds[1],
                "xmax_proj": bounds[2],
                "ymax_proj": bounds[3],
                "width_px": width,
                "height_px": height,
                "n_pixels": height * width,
                "geometry": bounding_box_wgs84.geometry[0],
                # "geometry_proj": bounding_box.geometry[0]
            }
        )

    zones = gpd.GeoDataFrame(pd.DataFrame(zones_list), crs=4326)

    non_covered_glaciers = rgi[~rgi.within(zones.dissolve().geometry[0])]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*empty DataFrame to file.*")
        non_covered_glaciers.to_file(
            cache_path.with_stem(cache_path.stem.replace("glacier_zones", "glacier_zones-missing_glaciers")),
            driver="GeoJSON",
        )

    if non_covered_glaciers.shape[0] > 0:
        print(
            f"Zones miss {non_covered_glaciers.shape[0]} / {rgi.shape[0]} glaciers above the area threshold ({area_threshold}"
            " km²) "
        )

    zones.to_file("zones.geojson", driver="GeoJSON")

    return zones


def test_get_best_utm_zone() -> None:
    assert get_best_utm_zone(15) == 33
    assert get_best_utm_zone(1) == 31
    assert get_best_utm_zone(-179.99) == 1
    assert get_best_utm_zone(169) == 59
    assert get_best_utm_zone(179.99) == 60
