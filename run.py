import matplotlib.pyplot as plt


def main() -> None:

    import surgedetection.main

    ids = {"fridtjov": "G014442E77835N", "nathorst": "G016633E77290N", "tuna": "G017497E78572N"}
    # import surgedetection.inputs.itslive
    # surgedetection.inputs.itslive.read_files()
    stack = surgedetection.main.make_glacier_stack(ids["tuna"])

    print(stack.sel(source="ASAR").dropna("time", how="all"))

    periglacial_vals = stack["sar_backscatter"].where(~stack["rgi_mask"])
    periglacial_med = periglacial_vals.median(["easting", "northing", "source"]).dropna("time")
    # periglacial_std = periglacial_vals.std(["easting", "northing", "source"]).dropna("time")
    sar = stack["sar_backscatter"].where(stack["rgi_mask"]).median(["easting", "northing", "source"]).dropna("time")
    sar /= periglacial_med

    vel = (
        stack["ice_velocity"]
        .where(stack["rgi_mask"])
        .dropna("source", how="all")
        .median(["easting", "northing", "source"])
    )

    # dhdt = stack["dhdt"].where(stack["rgi_mask"]).median(["easting", "northing", "source"]).dropna("time")

    plt.subplot(211)
    sar.plot.scatter()

    plt.subplot(212)
    vel.plot.scatter()

    # plt.subplot(313)
    # dhdt.plot()

    plt.show()
    # plt.show()
    # stack["sar_backscatter"].dropna("time", how="all").isel(time=0).dropna("source", how="all").isel(source=0).plot()
    # plt.show()


if __name__ == "__main__":
    main()
