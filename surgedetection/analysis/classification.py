import surgedetection.analysis.aggregation
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def main():
    # A massive lightup may be when the glacier decelerates (see Strongbreen, Penck)

    agg = surgedetection.analysis.aggregation.aggregate_region()
    stack = surgedetection.analysis.aggregation.interpret_stack()

    ids = {
        "nathorst": "RGI60-07.00322",
        "scheele": "RGI60-07.00283",
        "arnesen": "RGI60-07.00276",
        "tuna": "RGI60-07.01458",
        "monaco": "RGI60-07.01494",
        "krone": "RGI60-07.01464",
        "comfort": "RGI60-07.00511",
        "wahlen": "RGI60-07.00465",
        "negri": "RGI60-07.01506",
        "slak": "RGI60-07.00344",
        "strong": "RGI60-07.00296",
        "sko": "RGI60-07.00280",
        "paula": "RGI60-07.01470",
        "penck": "RGI60-07.00241",
        "recherche": "RGI60-07.00228",
        "storis": "RGI60-07.00027",
        "chyden": "RGI60-07.01500",
        "kongs": "RGI60-07.01481",
        "vonpost": "RGI60-07.01454",
        "ganskij": "RGI60-07.00897",
        "liestol": "RGI60-07.01472",
        "fridtjov": "RGI60-07.01100",
        "aavats": "RGI60-07.00501",
        "kongsbr": "RGI60-07.01482",
        "blomstr": "RGI60-07.00558",
        "fjortende": "RGI60-07.01492",
        "bodley": "RGI60-07.00042",
        "hayes": "RGI60-07.01479",
        "hamberg": "RGI60-07.01425",
        # "svalis": "RGI60-07.00440",
        "zawad": "RGI60-07.00235",
        "mendel": "RGI60-07.01396",
        "hinlopen": "RGI60-07.01559",
        "emma": "RGI60-07.00777",
        "sonklar": "RGI60-07.00892",
        "backlund": "RGI60-07.00890",
        "lillieh": "RGI60-07.00661",
        "vasili": "RGI60-07.00299",
        "sveits": "RGI60-07.00233",
        "polakk": "RGI60-07.00237",
        "dobrow": "RGI60-07.00298",
        "osborne": "RGI60-07.00482",
        "stone": "RGI60-07.01554",
        "osborneE": "RGI60-07.00476",
        "indrebo": "RGI60-07.00294",
    }
    #rgi = ids[list(ids.keys())[-1]]
    name = "arnesen"
    rgi = ids[name]
    p_scopes = {p: f"p_{p - 5}_{p + 5}" for p in range(5, 105, 10)}

    agg = agg.sel(rgi_id=rgi).sel(scope=list(p_scopes.values()))#.dropna("year", how="all")
    #bounds = stack["bboxes"].sel(rgi_id=rgi).values.ravel()
    buffer = 3000
    #bounds[[2, 3]] += buffer
    #bounds[[0, 1]] -= buffer
    #stack = stack.sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[3], bounds[1]))


    if False:
        plt.subplot(121)
        stack["ice_velocity"].sel(year=1992).plot()
        plt.subplot(122)
        stack["glacier_mask"].plot()
        plt.show()
    

    plot_params = {
        "sar_backscatter_diff": {
            "interval_length": 1,
            "cmap": "cividis",
            "vmax": 2,
        },
        "dhdt": {
            "interval_length": 5,
            "cmap": "bwr_r",
            "vmax": 2,
        },
        "dhdt2": {
            "interval_length": 10,
            "cmap": "bwr_r",
            "vmax": 0.5,
        },
        "ice_velocity": {
            "interval_length": 1,
            "cmap": "autumn_r",
            "vmin": 0,
            "vmax": 300,
        },
        "ice_velocity2": {
            "interval_length": 2,
            "cmap": "rainbow",
            "vmax": 100,
        }
    }

    #yr = stack.sel(year=2018).where(~stack["glacier_mask"])
    #plt.scatter(yr["dem"].values.ravel(), yr["sar_backscatter_diff"].values.ravel(), alpha=0.1, edgecolors="none")
    #plt.show()
    #print(stack)
    #return
    fig = plt.figure()
    plt.suptitle(name)
    aspect = 6

    variables = ["dhdt", "dhdt2", "sar_backscatter_diff", "ice_velocity", "ice_velocity2"]
    
    for i, variable_name in enumerate(variables):
        plt.subplot2grid((len(variables), aspect), (i, 0), colspan=aspect - 1)
        variable = agg[variable_name].dropna("year", how="all")
        vmin = plot_params[variable_name].get("vmin", -plot_params[variable_name]["vmax"])
        cmap = mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin, plot_params[variable_name]["vmax"], clip=False), cmap=plot_params[variable_name]["cmap"])
        arr = cmap.to_rgba(variable.values.T)

        err_var = f"{variable_name}_err"
        if err_var in agg.data_vars:
            err_arr = agg[err_var].sel(year=variable.year).values.T
            err_arr = np.clip(1.3 - (err_arr / np.nanmax(err_arr)), 0, 1)
            for j in range(3):
                arr[:, :, j] *= err_arr

        plt.imshow(
            arr,
            extent=[variable["year"].min() - plot_params[variable_name]["interval_length"], variable["year"].max(), 100, 0], 
            aspect="auto",
        )
        plt.ylim(0, 100)
        plt.xlim(agg["year"].min(), agg["year"].max())

        #if i < (len(variables) - 1):
        #    xticks = plt.gca().get_xticks()
        #    plt.xticks(xticks, labels=[""] * len(xticks))
        plt.subplot2grid((len(variables), aspect), (i, aspect -1))
        plt.axis("off")
        cbar = plt.colorbar(cmap, fraction=.4, aspect=3)
        cbar.set_label(variable_name)
        

    #agg.ice_velocity.plot()
    plt.tight_layout()
    plt.show()
