import geopandas as gpd
import numpy as np
import pandas as pd

# import surgedetection.inputs.rgi


def read_kaab_inventory():
    surges = (
        pd.read_excel("data/surge_inventories/surge_table_varya_final2.xlsx", sheet_name=1, header=None)
        .rename(
            columns={
                0: "glims_id",
                1: "rgi_id",
                2: "longitude",
                3: "latitude",
                4: "certainty",
                5: "start_year",
                6: "end_year",
                8: "name",
                11: "comment",
            }
        )
        .dropna(subset="start_year")
    )
    surges = surges[[col for col in surges.columns if isinstance(col, str)]]
    surges = surges[surges["certainty"] == 1]
    surges.loc[surges["end_year"] > 2022, "end_year"] = np.nan
    surges.loc[surges["start_year"] < 1500, "start_year"] = np.nan
    surges["source"] = "Kaab et al., 2023"

    return surges


def read_sevestre_inventory():

    surges = (
        pd.read_excel("data/surge_inventories/ORIGINAL Sevestre surge database Svalbard.xlsx")
        .query("Harmonised_Surge_Index == 3")
        .rename(
            columns={
                "Surge_onset": "start_year",
                "Surge_termination": "end_year",
                "Glacier_Name": "name",
            }
        )
    )
    surges["source"] = "Sevestre et al., 2015"


def read_guillet_inventory():

    surges = pd.read_csv("data/surge_inventories/gregguillet-HMA_STG_inventory-ee662aa/surge_inventory.csv").rename(
        columns={"RGIId": "rgi_id", "Name": "name"}
    )

    surge_cols = []
    for col in surges.columns:
        if "surge_idx_" not in col:
            continue
        surge_cols.append(col)

    surges["start_year"] = surges[surge_cols].T.idxmax().str.replace("surge_idx_", "").astype(float)
    surges["source"] = "Guillet et al., 2022"

    return surges


def read_mannerfelt_inventory():

    surges = pd.read_csv("manual_input/manual_surges.csv").rename(
        columns={"increase_start": "start_year", "end": "end_year"}
    )

    surges["source"] = "Mannerfelt et al., in prep."

    return surges


def read_all():

    surges = pd.concat(
        [
            read_guillet_inventory(),
            read_kaab_inventory(),
            read_sevestre_inventory(),
            read_mannerfelt_inventory(),
        ]
    )

    return surges[["rgi_id", "start_year", "end_year", "name", "source"]]


def main():

    # sevestre = read_sevestre_inventory()

    surges = read_all()
    print(surges)

    return
    kaab = read_kaab_inventory()

    # rgi = gpd.read_feather(".cache/read_all_rgi6_2f9813bcde2fe5377c47128cf3441d5052feea43.feather")
