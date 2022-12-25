import os
import shutil
import time
import urllib
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pyproj import CRS
from tqdm import tqdm

import surgedetection.cache
import surgedetection.io
import surgedetection.rasters
from surgedetection.constants import CONSTANTS
from surgedetection.rasters import RasterInput

BASE_DOWNLOAD_URL = "https://services-theia.sedoo.fr/glaciers/data/v1_0/"
YEARS = list(range(2000, 2020, 5))
RGI_ZONES = list(range(1, 20))

DATA_DIR = CONSTANTS.data_path.joinpath("hugonnet-etal-2021")


def get_filepaths(tarfile_dir: Path = DATA_DIR, crs: int | CRS = 32633) -> list[RasterInput]:

    full_tarfile_dirpath = CONSTANTS.data_path.joinpath(tarfile_dir)

    if isinstance(crs, int):
        crs = CRS.from_epsg(crs)

    rasters = []
    for filepath in full_tarfile_dirpath.glob("*.tar"):
        region = filepath.stem.split("_")[0]
        start_date = pd.to_datetime(filepath.stem.split("_")[-2])
        end_date = pd.to_datetime(filepath.stem.split("_")[-1])

        for kind in ["dhdt", "dhdt_err"]:
            rasters.append(
                RasterInput(
                    source="hugonnet-etal-2021",
                    start_date=start_date,
                    end_date=end_date,
                    kind=kind,
                    region=region,
                    filepath=load_tarfile(filepath, crs, pattern=".*" + kind + r"\.tif"),
                    multi_date=True,
                    multi_source=False,
                    time_prefix="dhdt",
                )
            )

    return rasters


def load_tarfile(
    filepath: Path,
    crs: CRS,
    pattern: str = r".*\.tif",
) -> Path:
    cache_filename = surgedetection.cache.get_cache_name("load_tarfile", args=[filepath, pattern, crs]).with_suffix(
        ".vrt"
    )

    if cache_filename.is_file():
        return cache_filename

    files = surgedetection.io.list_tar_filepaths(filepath, pattern=pattern, prepend_vsitar=True)

    surgedetection.rasters.merge_raster_tiles(
        filepaths=files,
        crs=crs,
        out_path=cache_filename,
    )

    return cache_filename


def aster_rgi_zone_mapping(rgi_zone: int) -> str:
    if rgi_zone not in [1, 2, 13, 14, 15]:
        return str(rgi_zone).zfill(2)
    if rgi_zone in [1, 2]:
        return "01_02"
    return "13_14_15"


def build_aster_url(path: str, args_dict: dict[str, Any] | None = None) -> str:
    """

    Modified from https://stackoverflow.com/a/44552191
    """
    url_parts = list(urllib.parse.urlparse(BASE_DOWNLOAD_URL))
    url_parts[2] += path
    if args_dict is not None:
        url_parts[4] = urllib.parse.urlencode(args_dict)
    return urllib.parse.urlunparse(url_parts)


def get_aster_request_url(rgi_region: str, period: str) -> str:
    response = requests.get(build_aster_url(f"prepare/{rgi_region}/{period}"))

    if response.status_code != 200:
        raise ValueError(response, response.content)

    request_id = response.content.decode()
    for i in range(20):

        response = requests.get(build_aster_url("/check", {"requestid": request_id}))
        if response.status_code != 200:
            raise ValueError(response, response.content)

        content = response.content.decode()
        if content == "DONE":
            break

        time.sleep(1)
    else:
        raise ValueError(f"Got unexpected response. Expected: 'DONE', got: {response.content}")

    return build_aster_url(f"/download/{request_id}")


def download_aster(rgi_region: str, period: str, filepath: Path) -> Path:
    url = get_aster_request_url(rgi_region=rgi_region, period=period)

    os.makedirs(filepath.parent, exist_ok=True)

    with requests.get(url, stream=True) as request:
        with open(filepath, "wb") as outfile:
            shutil.copyfileobj(request.raw, outfile)

    return filepath


def download_all_aster() -> None:
    rgi_queries = {aster_rgi_zone_mapping(zone) + "_rgi60" for zone in RGI_ZONES}
    period_queries = [f"{year}-01-01_{year + 5}-01-01" for year in YEARS]

    queries = []
    for rgi in rgi_queries:
        for period in period_queries:
            local_filepath = DATA_DIR.joinpath(f"{rgi}_{period}.tar")
            if local_filepath.is_file():
                continue
            queries.append({"rgi_region": rgi, "period": period, "filepath": local_filepath})

    for query in tqdm(queries, desc="Downloading ASTER data"):
        download_aster(**query)
        # List the filepaths in the tarfile. If the file is not a tarfile, this will fail
        surgedetection.io.list_tar_filepaths(query["filepath"])
