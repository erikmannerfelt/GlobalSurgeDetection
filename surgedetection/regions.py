import geopandas as gpd
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


def make_glacier_regions(buffer: float = 5000.0, mod: float = 1000.0, area_threshold: float = 1.6) -> gpd.GeoDataFrame:
    manual_entries = parse_manual_glacier_zones()  # .query("name == 'NE Russia'")

    cache_path = surgedetection.cache.get_cache_name(
        "make_glacier_regions", [buffer, mod, area_threshold], extension="geojson"
    )

    if cache_path.is_file():
        zones = gpd.read_file(cache_path)
        zones["crs"] = zones["crs_epsg"].apply(pyproj.CRS.from_epsg)
        return zones
    rgi_o2_regions = surgedetection.inputs.rgi.read_rgi6_regions()

    # All glaciers with fewer than 4x4 pixels are excluded (400 * 400 m == 1.6 km²)
    rgi = surgedetection.inputs.rgi.read_all_rgi6(query=f"Area > {area_threshold}")

    excluded_antarctica_zones = [str(v) for v in (list(range(11, 19)) + list(range(22, 25)))]

    rgi.drop(index=rgi[(rgi["O1Region"] == "19") & rgi["O2Region"].isin(excluded_antarctica_zones)].index, inplace=True)
    # rgi.drop(rgi[((rgi["O1Region"] == "19") & rgi["O2Region"].isin([excluded_antarctica_zones])), inplace=True)
    rgi["O1Region"] = rgi["O1Region"].astype(int)

    zones_list = []
    # rgi_outlines = {}
    for _, entry in tqdm(
        manual_entries.iterrows(), total=manual_entries.shape[0], desc="Calculating optimal glacier regions"
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
        height = int((bounds[3] - bounds[1]) / CONSTANTS.pixel_size)
        width = int((bounds[2] - bounds[0]) / CONSTANTS.pixel_size)

        bounding_box_wgs84 = generate_safe_wgs_bbox(bounds, outlines.crs)

        lon_vertices, lat_vertices = bounding_box_wgs84.geometry[0].exterior.xy

        center_lon = round(np.mean(lon_vertices))
        center_lat = round(np.mean(lat_vertices))

        lon_width = round(np.max(lon_vertices) - np.min(lon_vertices))
        lat_width = round(np.max(lat_vertices) - np.min(lat_vertices))

        if any(val > 90 for val in (lon_width, lat_width)):
            raise ValueError(f"Something is wrong with the box width for {entry=}: {lon_width=}, {lat_width=}")

        region_id = "REG{lat_ref}{lat}{lon_ref}{lon}X{lon_width}Y{lat_width}".format(
            lon_ref="E" if center_lon >= 0 else "W",
            lon=str(abs(center_lon)).zfill(3),
            lat_ref="N" if center_lat >= 0 else "S",
            lat=str(abs(center_lat)).zfill(2),
            lon_width=str(lon_width).zfill(2),
            lat_width=str(lat_width).zfill(2),
        )
        label = (
            region_id
            + "-"
            + "".join(
                map(
                    lambda s: s if s[0].isupper() else s.capitalize(),  # type: ignore
                    entry["name"].replace("/", "And ").replace("&", "And").replace("  ", " ").split(" "),
                )
            )
        )

        zones_list.append(
            {
                "name": entry["name"],
                "region_id": region_id,
                "label": label,
                "O1_regions": "/".join(map(str, entry["O1_regions"])),
                "O2_regions": "/".join(map(str, entry["O2_regions"])),
                "rgi_regions": "/".join(entry["rgi_regions"]),
                "crs_epsg": entry["crs"].to_epsg(),
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

    zones["creation_date"] = pd.Timestamp.now(tz="utc").replace(microsecond=0).isoformat()

    non_covered_glaciers = rgi[~rgi.within(zones.dissolve().geometry[0])]
    if non_covered_glaciers.shape[0] > 0:
        print(
            f"Zones miss {non_covered_glaciers.shape[0]} / {rgi.shape[0]} "
            f"glaciers above the area threshold ({area_threshold} km²)."
        )
        non_covered_glaciers.to_file(
            cache_path.with_stem(cache_path.stem.replace("glacier_regions", "glacier_regions-missing_glaciers")),
            driver="GeoJSON",
        )

    zones.to_file(cache_path, driver="GeoJSON")

    surgedetection.cache.symlink_to_output(cache_path, "shapes/glacier_regions")
    zones["crs"] = zones["crs_epsg"].apply(pyproj.CRS.from_epsg)

    return zones


def test_get_best_utm_zone() -> None:
    assert get_best_utm_zone(15) == 33
    assert get_best_utm_zone(1) == 31
    assert get_best_utm_zone(-179.99) == 1
    assert get_best_utm_zone(169) == 59
    assert get_best_utm_zone(179.99) == 60
