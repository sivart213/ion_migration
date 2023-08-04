# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 22:48:24 2022

@author: j2cle
"""

import numpy as np
import pandas as pd

# from utilities import np_ratio, poisson, Time, Length, Temp

from research_tools.equations import screened_permitivity, np_ratio, poisson
from research_tools.functions import get_const, convert_val

surf = 5e19

time = convert_val(24, "h", "s")
times = convert_val(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]), "h", "s")

temp = convert_val(80, "C", "K")

dif = 1e-15
difs = [1e-16, 1e-15, 1e-14]

thick = convert_val(450, "um", "cm")

depth = convert_val(15, "um", "cm")
depths = np.linspace(0, depth, 1000)
depths[0] = 1e-30

valence = 0.5
valences = [0.01, 0.1, 1]

volt = 1500

df = pd.DataFrame(
    [
        dict(
            time=time,
            depth=depths,
            thick=thick,
            temp=temp,
            efield=volt / thick * valence,
            dif=dif,
        )
    ]
)
df_dict = dict(
    time=time,
    depth=depths,
    thick=thick,
    temp=temp,
    voltage=volt,
    dif=dif,
    valence=valence,
)
res_df = pd.DataFrame()
res_df["depth"] = convert_val(depths, "cm", "um")


if 1:
    res_df[f"conc"] = surf * np_ratio(df_dict)
    # res_df[f"conc"] = surf * np_ratio(time, df, pd.DataFrame(), target="time", scale="lin")
    res_df[f"volt1"] = (
        poisson(
            rel_perm=2.95, valence=valence, thick=thick, conc=res_df[f"conc"].to_numpy()
        )
        / thick
        * depths
    )
    res_df[f"volt2"] = (
        poisson(
            rel_perm=2.95,
            valence=valence,
            thick=thick,
            conc=res_df[f"conc"].cumsum().to_numpy(),
        )
        / thick
        * depths
    )
    res_df[f"volt3"] = (
        poisson(rel_perm=2.95, valence=valence, thick=thick, conc=res_df[f"conc"].max())
        / thick
        * depths
    )

if 0:
    for t in times:
        df["time"] = t
        res_df[f"conc_{t/3600}"] = surf * np_ratio(
            t, df, pd.DataFrame(), target="time", scale="lin"
        )
        # res_df[f"volt_{t/3600}"] = poisson(2.65, thick=thick, conc=res_df[f"conc_{t/3600}"].to_numpy()) / thick * depths
        res_df[f"volt_{t/3600}"] = (
            get_const("elementary_charge")
            * thick
            * res_df[f"conc_{t/3600}"].to_numpy()
            / (2 * 2.65 * get_const("e0", False, ["farad", "cm"]))
        )

if 0:
    for t in times:
        df["time"] = t
        res_df[f"conc_{t/3600}"] = surf * np_ratio(
            t, df, pd.DataFrame(), target="time", scale="lin"
        )
        # res_df[f"volt_{t/3600}"] = poisson(2.65, thick=thick, conc=res_df[f"conc_{t/3600}"].to_numpy()) / thick * depths
        res_df[f"volt_{t/3600}"] = (
            get_const("elementary_charge")
            * thick
            * res_df[f"conc_{t/3600}"].to_numpy()
            / (2 * 2.65 * get_const("e0", False, ["farad", "cm"]))
        )


# volts = volt - volt/thick * depths
# # poisson_curve = poisson(2.65, volt=1500, thick=thick) * thick / depths
# # poisson_curve = 2 * 2.65 * get_const("e0", False, ["farad", "cm"]) * volt / (get_const("elementary_charge") * thick * depths)
# poisson_curve = 2 * 2.65 * get_const("e0", False, ["farad", "cm"]) * volt / (0.01 * get_const("elementary_charge") * thick * depths)
# poisson_curve = 2 * 2.65 * get_const("e0", False, ["farad", "cm"]) * (volt/thick*depths) / (get_const("elementary_charge") * depths**2)
# poisson_curve = -2 * 2.65 * get_const("e0", False, ["farad", "cm"]) * volts / (get_const("elementary_charge") * (depths / thick - 2))
# poisson_curve = -2 * 2.65 * get_const("e0", False, ["farad", "cm"]) * volts / (get_const("elementary_charge") * (depths ** 2 - 2 * depths * thick))

# efield = get_const("elementary_charge") * Length(4.5, "um").cm * 1e14 / (2 * 2.65 * get_const("e0", False, ["farad", "cm"]))
# p_depths = Length(2 * 2.65 * get_const("e0", False, ["farad", "cm"]) * volt / (get_const("elementary_charge") * 1e14), "cm").um
