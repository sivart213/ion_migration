# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:03:21 2022.

@author: j2cle
"""

import re
import numpy as np
import pandas as pd

from defect_code.equations import Statistics
from defect_code.functions import convert_val, f_find, sample_array


# %%
def h5_bulk_in(
    file_pth, x_size=100, x_max=500, **kwargs
):  # t_step=1800, t_size=10, t_max=10800.0,
    files = [f for f in f_find(file_pth, file_filter=".h5")]

    file_dict = {"conc": {}, "volt": {}, "attrs": {}, "key_error": {}}
    for f_path in files:
        file = load(f_path, target="", file_filter="h5")
        try:
            # parse desired datasets
            atr = file[1].get("EVA", file[1].get("L1", {}))
            data = file[0].get("EVA", file[0].get("L1", {}))
            times = np.unique(np.array(file[0].get("time", [])))
            if len(times) == 0:
                times = np.array(
                    [
                        int(v["time"])
                        for v in atr["concentration"].values()
                        if isinstance(v, dict)
                    ]
                )

            atr["time"] = times.max()
            attr_flux = [k for k in atr.keys() if re.search(".*flux.+", k)]
            for flux in attr_flux:
                flux_n = re.search(".*flux", flux).group()
                atr[f"{flux_n}_type"] = (
                    flux.replace(flux_n, "").replace("_", " ").strip()
                )
                atr[flux_n] = atr.pop(flux)

            file_dict["attrs"][f_path.stem] = pd.Series(atr)
            times.sort()

            tlist = sample_array(
                times,
                arr_max=kwargs.get("t_max", max(times)),
                arr_step=kwargs.get("t_step"),
                arr_size=kwargs.get("t_size", len(times)),
            )

            # parse desired indexes
            xconv = 1e4
            if np.log10(max(data["x"])) >= -1.1:
                xconv = 1
            xconv = kwargs.get("xconv", xconv)

            full_x = data["x"] * xconv
            data_x, data_n = sample_array(
                full_x, get_index=True, arr_size=x_size, arr_max=x_max
            )

            # get desired data points
            conc = {
                int(convert_val(v["time"], "s", "h")): data["concentration"][k][data_n]
                for k, v in atr["concentration"].items()
                if isinstance(v, dict) and v["time"] in tlist
            }
            pot = {
                int(convert_val(v["time"], "s", "h")): data["potential"][k][data_n]
                for k, v in atr["potential"].items()
                if isinstance(v, dict) and v["time"] in tlist
            }

            # convert to df
            conc = pd.DataFrame(conc, index=data_x, dtype=float)
            conc = conc.reindex(sorted(conc.columns), axis=1)
            pot = pd.DataFrame(pot, index=data_x, dtype=float)
            pot = pot.reindex(sorted(pot.columns), axis=1)

            file_dict["conc"][f_path.stem] = conc
            file_dict["volt"][f_path.stem] = pot
        except KeyError:
            file_dict["key_error"][f_path.stem] = file

    file_dict["attrs"] = pd.DataFrame(file_dict["attrs"]).T

    return file_dict


def compare(data, files, target="conc", col=24, stat_type="r2"):
    res_arr = [data[comp][target][col].to_numpy() for comp in files]
    stats = Statistics(res_arr[0], res_arr[1])
    return stats[stat_type]


def compile_attrs(df):
    attr_pairs = [
        ["T", "temp_c"],
        ["V", "stress_voltage"],
        ["E", "electric_field_eff"],
        ["z", "valency", "valence", "ion_valency"],
        ["C_bulk", "Cbulk"],
        ["S_0", "surface_concentration", "csurf", "c_surf"],
        ["C_0", "volume_concentration", "csurf_vol", "c_surf_vol"],
        ["k", "rate", "k_0", "flux"],
        ["screen", "screen_leng"],
    ]

    if isinstance(df.get("in_flux", None), (float, int)):
        attr_pairs.append(["k", "in_flux"])

    attr_drop = [
        "drift_velocity",
        "electric_field_app",
        "t_max",
        "t_final",
        "concentration",
        "potential",
    ]

    for pair in attr_pairs:
        df = df.groupby(
            np.array([c if c not in pair else pair[0] for c in df.columns]), axis=1
        ).first()

    for to_go in attr_drop:
        if to_go in df.columns:
            df = df.drop(columns=to_go)

    df1 = df.infer_objects()
    if "notes" in df.columns:
        df1["notes"] = df["notes"].astype(str)

    df1["time"] = [convert_val(t, "s", "h") for t in df1["time"]]
    # df1["sim_depths"] = [
    #     value["conc"].index.max() for value in data.values() if "conc" in value.keys()
    # ]
    # df1["Cmax"] = [
    #     value["conc"].max().max() for value in data.values() if "conc" in value.keys()
    # ]
    return df1


def originprint(df):
    if isinstance(df, pd.DataFrame):
        for ind in df.index:
            print(f"Sheet {ind}")
            originprint(df.loc[ind, :])
            print("")
        return
    df = df.dropna()
    print(
        "mPNP\n{0:d} h,\n{1:d} \\+(o)C,\n{2:.1f} kV,\nr{3},".format(
            int(df.get("time", 0)),
            int(df.get("T", 0)),
            float(df.get("V", 0) * 1e-3),
            df.name,
        )
    )

    print(
        "e\\-(r)={0:.2f},\nz={1:.1f},\n\\g(l)\\-(d)={2:g}nm,\nD={3:.2E},".format(
            float(df.get("er", 0)),
            float(df.get("z", 1)),
            float(df.get("screen", 0)),
            float(df.get("D", 0)),
        )
    )

    if isinstance(df.get("in_flux", None), (list, tuple, np.ndarray)):
        print(
            "S\\-(0)={0:.2E},\nh={1:.1E}, m={2:.0E},".format(
                float(df.get("C_0", df.get("S_0", 1))),
                float(df["in_flux"][0]),
                float(df["in_flux"][1]),
            )
        )
    else:
        print(
            "S\\-(0)={0:.2E},\nk={1:.1E},".format(
                float(df.get("S_0", df.get("C_0", 1))),
                float(df.get("k", 0)),
            )
        )
    return


# %% Operations
if __name__ == "__main__":
    from defect_code.functions import save, lineplot_slider, p_find, load

    folder = "m80r7"  #  "m60r8", "m80r4", "mNoEr4"

    data_pth = p_find(
        "Dropbox (ASU)",
        "Work Docs",
        "Data",
        "Analysis",
        "Simulations",
        "PNP",
        "EVA",
        base="home",
    )

    if 1:
        eva_dict = h5_bulk_in(
            data_pth / folder,
            x_size=500,
            x_max=5,
            # t_step=convert_val(6, "h", "s"),
            t_size=6,
            # t_max=convert_val(1, "day", "s"),
        )

        eva_attr = compile_attrs(eva_dict["attrs"])
        save(eva_dict["conc"], data_pth / folder, f"{folder}_conc")
        save(eva_dict["volt"], data_pth / folder, f"{folder}_volt")

    if 1:
        print(folder)
        originprint(eva_attr)

    if 0:
        res = eva_dict
        title = "Simulation diff Comparison"
        name1 = "surf_high_base"  # "eva_sim_3\eva_7"
        name2 = "surf_high_diff_high"  # "eva_sim_5\eva_1"

        match = compare(res, [name1, name2], target="conc", col=6)

        lineplot_slider(
            res[name1]["conc"],
            xlimit=[0, 5],
            yscale="log",
            ylimit=[1e10, res[name1]["conc"].max().max()],
            name=title,
            xname="Depth (um)",
            yname="[Na] (cm-3)",
            data2=res[name2]["conc"],
        )

        lineplot_slider(
            res[name1]["volt"],
            xlimit=[0, 4],
            ylimit=[1475, 1510],
            name=title,
            xname="Depth (um)",
            yname="Voltage (V)",
            data2=res[name2]["volt"],
        )
