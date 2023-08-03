# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:42:38 2021

@author: j2cle
"""
# %%
import numpy as np
import pandas as pd
import General_functions as gf
import Units_Primary4 as up

# import matplotlib.pyplot as plt
# from matplotlib import ticker
from scipy.special import erfc
from scipy import optimize

# from matplotlib.colors import LogNorm
# from scipy.optimize import curve_fit, fsolve, minimize
# from functools import partial
from dataclasses import dataclass, field
from dataclasses import InitVar


def arrh(T, pre_fac, E_A):
    return pre_fac * np.exp(E_A / (gf.KB_EV * T))


def c_ratio_np(depth, thick, temp, e_app, time, diff, mob):
    term_B = erfc(-mob * e_app * time / (2 * np.sqrt(diff * time)))
    return (1 / (2 * term_B)) * (
        erfc((depth - mob * e_app * time) / (2 * np.sqrt(diff * time)))
        + erfc(-(depth - 2 * thick + mob * e_app * time) / (2 * np.sqrt(diff * time)))
    )


def c_ratio_np_df(time, df):  # thick,temp,e_app,time,diff,mob):
    mob = df["mob"].to_numpy()
    e_app = df["efield"].to_numpy()
    depth = df["depth"].to_numpy()
    diff = df["mob"].to_numpy()
    thick = df["thick"].to_numpy()
    term_B = erfc(-mob * e_app * time / (2 * np.sqrt(diff * time)))
    return (1 / (2 * term_B)) * (
        erfc((depth - mob * e_app * time) / (2 * np.sqrt(diff * time)))
        + erfc(-(depth - 2 * thick + mob * e_app * time) / (2 * np.sqrt(diff * time)))
    )


def c_ratio_np_s(time, df):  # thick,temp,e_app,time,diff,mob):
    mob = df["mob"]
    e_app = df["efield"]
    depth = df["depth"]
    diff = df["mob"]
    thick = df["thick"]
    term_B = erfc(-mob * e_app * time / (2 * np.sqrt(diff * time)))
    return (1 / (2 * term_B)) * (
        erfc((depth - mob * e_app * time) / (2 * np.sqrt(diff * time)))
        + erfc(-(depth - 2 * thick + mob * e_app * time) / (2 * np.sqrt(diff * time)))
    )


def find_tauc_np_cratio_3d(
    time, depth, temp, e_app, d_0, e_a, target=0.08, thick=450e-4
):
    diff = d_0 * np.exp(-e_a / (gf.KB_EV * temp))
    mob = diff / (gf.KB_EV * temp)
    if depth == 0:
        depth = (2 * np.sqrt(diff * time)) + mob * e_app * time
    if thick == 0:
        thick = (2 * np.sqrt(diff * time)) + mob * e_app * time
    ratio = c_ratio_np(depth, thick, temp, e_app, time, diff, mob)
    if isinstance(ratio, float):
        if ratio < 1e-10:
            ratio = 1
    else:
        for n in range(len(ratio)):
            if ratio[n] < 1e-10:
                ratio[n] = 1
    return (ratio - target) ** 2


def find_tauc_np_cratio_3d_df(time, df, target=0.08):
    ratio = c_ratio_np_s(time, df)
    if isinstance(ratio, float):
        if ratio < 1e-10:
            ratio = 1
    else:
        for n in range(len(ratio)):
            if ratio[n] < 1e-10:
                ratio[n] = 1
    return (ratio - target) ** 2


def find_tauc_np_cratio_3d_alt(
    time, depth, temp, e_app, d_0, e_a, target=0.08, thick=450e-4
):
    diff = d_0 * np.exp(-e_a / (gf.KB_EV * temp))
    mob = diff / (gf.KB_EV * temp)
    if depth == 0:
        depth = (2 * np.sqrt(diff * time)) + mob * e_app * time
    if thick == 0:
        thick = (2 * np.sqrt(diff * time)) + mob * e_app * time
    ratio = c_ratio_np(depth, thick, temp, e_app, time, diff, mob)

    return ratio


def layer_btt(time, depth, temp, e_app, diff, mob, target=0.08, thick=450e-4):
    time = 10 ** time
    if depth == 0:
        depth = (2 * np.sqrt(diff * time)) + mob * e_app * time
    if thick == 0:
        thick = (2 * np.sqrt(diff * time)) + mob * e_app * time
    ratio = c_ratio_np(depth, thick, temp, e_app, time, diff, mob)
    if isinstance(ratio, float):
        if ratio < 1e-10:
            ratio = 1
    else:
        for n in range(len(ratio)):
            if ratio[n] < 1e-10:
                ratio[n] = 1
    return (ratio - target) ** 2


@dataclass
class Layer(object):
    """Return sum of squared errors (pred vs actual)."""

    # Invisible attribute (init-only)
    material: str = "undefined"
    thick: float = 0.045
    diff_type: InitVar[str] = "undefined"

    # Initialized attribute
    temp: float = 25
    efield: float = 0
    resistivity: float = field(init=False)

    # Generated attribute
    diff: float = field(init=False)
    mob: float = field(init=False)

    area: float = 1
    res: float = field(init=False)
    volt: float = field(init=False)
    cap: float = field(init=False)
    charge: float = field(init=False)

    topC: float = 1e21
    botC: float = 1e10
    btt: float = field(init=False)

    @property
    def thick(self) -> float:
        """Return thickness of layer."""
        return self._thick.cm

    @thick.setter
    def thick(self, val):
        if isinstance(val, up.Length):
            self._thick = val
        elif isinstance(val, property):
            self._thick = up.Length(0.045, "cm")
        elif isinstance(val, (tuple, list)):
            self._thick = up.Length(*val)
        elif isinstance(val, (dict)):
            self._thick = up.Length(**val)
        else:
            self._thick = up.Length(val, "cm")

    @property
    def temp(self) -> float:
        """Return thickness of layer."""
        return self._temp.C

    @temp.setter
    def temp(self, val):
        if isinstance(val, up.Temp):
            self._temp = val
        elif isinstance(val, property):
            self._temp = up.Temp(25, "C")
        elif isinstance(val, (tuple, list)):
            self._temp = up.Temp(*val)
        elif isinstance(val, (dict)):
            self._temp = up.Temp(**val)
        else:
            self._temp = up.Temp(val, "C")

    @property
    def area(self) -> float:
        """Return thickness of layer."""
        return self._area.cm

    @area.setter
    def area(self, val):
        if isinstance(val, up.Length):
            self._area = val
        elif isinstance(val, property):
            self._area = up.Length(1, "cm")
        elif isinstance(val, (tuple, list)):
            self._area = up.Length(*val)
        elif isinstance(val, (dict)):
            self._area = up.Length(**val)
        else:
            self._area = up.Length(val, "cm")

    @property
    def efield(self) -> float:
        """Return thickness of layer."""
        self._efield = up.Volt(self._volt.V / self._thick.cm)
        return self._efield.V

    @efield.setter
    def efield(self, val):
        if isinstance(val, property):
            self._volt = up.Volt(0, "V")
        else:
            self.volt = val * self._thick.cm

    @property
    def volt(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self._volt.V

    @volt.setter
    def volt(self, val):
        if isinstance(val, up.Volt):
            self._volt = val
        elif isinstance(val, property):
            self._volt = up.Volt(self.efield * self._thick.cm, "V")
        elif isinstance(val, (tuple, list)):
            self._volt = up.Volt(*val)
        elif isinstance(val, (dict)):
            self._volt = up.Volt(**val)
        else:
            self._volt = up.Volt(val, "V")

    @property
    def resistivity(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self._res_pre * np.exp(self._res_ea / (gf.KB_EV * self._temp.K))

    @resistivity.setter
    def resistivity(self, val):
        self._res_pre = val
        self._res_ea = 0

    @property
    def res(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        self._res = up.Res(self.resistivity * self._thick.cm / self._area.cm)
        return self._res.Ohm

    @res.setter
    def res(self, _):
        pass

    @property
    def diff(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self._d_0 * np.exp(-self._e_a / (gf.KB_EV * self._temp.K))

    @diff.setter
    def diff(self, _):
        pass

    @property
    def mob(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self.diff / (gf.KB_EV * self._temp.K)

    @mob.setter
    def mob(self, _):
        pass

    @property
    def cap(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self._er * self._area.cm / self._thick.cm

    @cap.setter
    def cap(self, _):
        pass

    @property
    def charge(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return self.cap * self._volt.V

    @charge.setter
    def charge(self, _):
        pass

    @property
    def btt(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_btt"):
            self.btt = {"jac": "3-point", "xtol": 1e-12}
        times = np.logspace(1, 12, 100)
        ratio_array_full = c_ratio_np(
            self.thick,
            self.thick,
            self._temp.K,
            self.efield,
            times,
            self.diff,
            self.mob,
        )
        ratio_array_local = np.where((ratio_array_full > 0) & (ratio_array_full < 1))[0]
        if len(ratio_array_local) == 0:
            x0 = times[ratio_array_full[::-1].argmin()]
            bounds = (
                times[ratio_array_full[::-1].argmin()],
                times[ratio_array_full.argmax()],
            )
        elif len(ratio_array_local) == 1:
            x0 = times[ratio_array_local]
            bounds = (times[ratio_array_local - 1][0], times[ratio_array_local + 1][0])
        elif len(ratio_array_local) == 2:
            x0 = times[ratio_array_local[0]]
            bounds = (times[ratio_array_local[0]], times[ratio_array_local[-1]])
        else:
            x0 = times[ratio_array_local[0]]
            bounds = (times[ratio_array_local[0]], times[ratio_array_local[-1]])
        vals = {"x0": np.log10(x0), "bounds": np.log10(bounds)}
        val = {**vals, **self._btt_vals}
        try:
            btt = (
                10
                ** optimize.least_squares(
                    layer_btt,
                    args=(
                        self.thick,
                        self._temp.K,
                        self.efield,
                        self.diff,
                        self.mob,
                        self.botC / self.topC,
                        self.thick,
                    ),
                    **val,
                ).x
            )
            self._btt = up.Time(btt)
        except ValueError:
            self._btt = up.Time(1)
        return self._btt.s

    @btt.setter
    def btt(self, val):
        if isinstance(val, property):
            pass
        else:
            self._btt_vals = val

    def __post_init__(self, diff_type):
        """Return layer object."""

        self._diff_type = diff_type.lower()
        try:
            self._res_pre = gf.mat_database.loc[self.material.lower(), "pre"]
        except KeyError:
            self._res_pre = gf.mat_database.loc["eva", "pre"]
        try:
            self._res_ea = gf.mat_database.loc[self.material.lower(), "ea"]
        except KeyError:
            self._res_ea = gf.mat_database.loc["eva", "ea"]
        try:
            self._er = gf.mat_database.loc[self.material.lower(), "perm"]
        except KeyError:
            self._er = gf.mat_database.loc["eva", "perm"]
        try:
            self._d_0 = gf.diff_arrh.loc[diff_type.lower(), "pre"]
        except KeyError:
            self._d_0 = gf.diff_arrh.loc["ave", "pre"]
        try:
            self._e_a = gf.diff_arrh.loc[diff_type.lower(), "ea"]
        except KeyError:
            self._e_a = gf.diff_arrh.loc["ave", "ea"]


class MatDatabase(object):
    def __init__(self):
        self.path = "C:\\Users\\j2cle\\Work Docs\\Data\\Databases"
        self.file = "material_data.xlsx"
        self.database = pd.read_excel(f"{self.path}\\{self.file}", index_col=[0, 1, 2])

    @property
    def material(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_material"):
            self._material = "air"
        return self._material

    @material.setter
    def material(self, val):
        if self.database.index.isin([val], level=0).any():
            self._material = val

    @property
    def mat_data(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.database.loc[self.material, :]

    def get_df(self, material, res=1, perm=1, diff=1, thick=1):
        self.material = material
        self.data = self.mat_data.xs(1, level="label").copy()
        labels = {"resis": res, "perm": perm, "diff": diff, "thick": thick}

        if not all(val == 1 for val in labels.values()):
            self.change_data(labels)
        return self.data

    def change_data(self, labels):
        for key, val in labels.items():
            if not isinstance(val, int):
                val = self.note_to_label(key, val)
            self.data.loc[key, :] = self.mat_data.loc[(key, val), :]

    def note_to_label(self, info, label):
        data = self.mat_data.loc[info, :]
        try:
            if isinstance(label, (list, tuple, np.ndarray)):
                for l in label:
                    data = data[data.note.str.contains(l)]
                return data.index[0]
            elif isinstance(label, str):
                return data[data.note.str.contains(label)].index[0]
            elif isinstance(label, float):
                if data.index.isin([int(label)]).any():
                    return int(label)
                else:
                    return 1
            else:
                return 1
        except IndexError:
            return 1

    def add(self, material, info, note="", var1=1, var2=1):
        label = self.database.loc[(material, info), :].index.max() + 1
        self.database.loc[(material, info, label), :] = [note, var1, var2]

    def save(self):
        self.database.to_excel(f"{self.path}\\{self.file}")


class Layer2(object):
    """Return sum of squared errors (pred vs actual)."""

    def __init__(
        self,
        material,
        data_guide={"resis": 1, "perm": 1, "diff": 1, "thick": 1},
        thick=0.045,
        area=1,
        temp=25,
        efield=0,
        topC=1e21,
        botC=1e10,
        btt_sim=False,
        ls_params={},
    ):
        """Return layer object."""
        # mat_test1=MatDatabase()
        # Invisible attribute (init-only)
        self.material = material
        self.data_guide = data_guide

        self.data_imp()

        self.thick = up.Length(self.data.var1["thick"], self.data.var2["thick"])

        # Initialized attribute
        self.temp = temp
        self.efield = efield

        # Generated attribute
        self.area = area

        self.topC = topC
        self.botC = botC

        self.btt = up.Time(1)

        if btt_sim:
            self.btt_sim(ls_params)

    @property
    def thick(self):
        """Return thickness of layer."""
        return up.Length(self.data.var1["thick"], self.data.var2["thick"])

    @thick.setter
    def thick(self, val):
        if isinstance(val, up.Length):
            self.data.var1["thick"] = float(str(val))
            self.data.var2["thick"] = val.print_unit
        elif isinstance(val, (tuple, list)):
            self.data.var1["thick"] = val[0]
            self.data.var2["thick"] = val[1]
        elif isinstance(val, (dict)):
            self.data.var1["thick"] = val["value"]
            self.data.var2["thick"] = val["unit"]
        else:
            self.data.var1["diff"] = val
            self.data.var2["diff"] = "cm"

    @property
    def temp(self):
        """Return thickness of layer."""
        return self._temp

    @temp.setter
    def temp(self, val):
        if isinstance(val, up.Temp):
            self._temp = val
        elif isinstance(val, (tuple, list)):
            self._temp = up.Temp(*val)
        elif isinstance(val, (dict)):
            self._temp = up.Temp(**val)
        else:
            self._temp = up.Temp(val, "C")

    @property
    def area(self):
        """Return thickness of layer."""
        return self._area

    @area.setter
    def area(self, val):
        if isinstance(val, up.Length):
            self._area = val
        elif isinstance(val, (tuple, list)):
            self._area = up.Length(*val)
        elif isinstance(val, (dict)):
            self._area = up.Length(**val)
        else:
            self._area = up.Length(val, "cm")

    @property
    def efield(self):
        """Return V/cm as volt type asuming cm's"""
        return self._efield

    @efield.setter
    def efield(self, val):
        """Ensures input is a unit class"""
        if isinstance(val, up.Volt):
            self._efield = val
        elif isinstance(val, (tuple, list)):
            self._efield = up.Volt(*val)
        elif isinstance(val, (dict)):
            self._efield = up.Volt(**val)
        else:
            self._efield = up.Volt(val, "V")

    @property
    def resistivity(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.data.var1["resis"] * np.exp(
            self.data.var2["resis"] / (gf.KB_EV * self.temp.K)
        )

    @resistivity.setter
    def resistivity(self, val):
        self.data.var1["resis"] = val
        self.data.var2["resis"] = 0

    @property
    def volt(self):
        """Return sum of squared errors (pred vs actual)."""
        return up.Volt(self.efield.V * self.thick.cm)

    @volt.setter
    def volt(self, val):
        """Sets efield as volt is calculated from efield """
        self.efield = val / self.thick.cm

    @property
    def res(self):
        """Return sum of squared errors (pred vs actual)."""
        return up.Res(self.resistivity * self.thick.cm / self._area.cm)

    @property
    def curr(self):
        """Return sum of squared errors (pred vs actual)."""
        return up.Curr(self.volt.V / self.res.Ohm)

    @property
    def cap(self):
        """Return sum of squared errors (pred vs actual)."""
        er = gf.tocm(gf.PERM, "m", inv=True) * self.data.var1["perm"]
        return er * self.area.cm / self.thick.cm

    @property
    def charge(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.cap * self.volt.V

    @property
    def diff(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.data.var1["diff"] * np.exp(
            -self.data.var2["diff"] / (gf.KB_EV * self.temp.K)
        )

    @diff.setter
    def diff(self, val):
        self.data.var1["diff"] = val
        self.data.var2["diff"] = 0

    @property
    def mob(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.diff / (gf.KB_EV * self.temp.K)

    @property
    def info(self):
        """Return the necessary information."""

        info = {
            "Material": self.material,
            "thickness (um)": self.thick.um,
            "temp (C)": self.temp.C,
            "efield (MV/cm)": self.efield.MV,
            "resistivity (ohm.cm)": self.resistivity,
            "voltage (V)": self.volt.V,
            "resistance (ohm)": self.res.Ohm,
            "current (A)": self.curr.A,
            "capacitance": self.cap,
            "charge": self.charge,
            "top conc": self.topC,
            "bot conc": self.botC,
            "BTT": self.btt.class_values()[
                np.log10(self.btt.class_values()) >= 0
            ].min(),
            "BTT unit": self.btt.class_values()[
                np.log10(self.btt.class_values()) >= 0
            ].idxmin(),
        }
        return pd.Series(info)

    def btt_sim(self, ls_params={}):
        """Return sum of squared errors (pred vs actual)."""
        if self.diff != 0:

            ls_params = {**{"jac": "3-point", "xtol": 1e-12}, **ls_params}

            times = np.logspace(1, 12, 100)
            ratio_array_full = c_ratio_np(
                self.thick.cm,
                self.thick.cm,
                self.temp.K,
                self.efield.V,
                times,
                self.diff,
                self.mob,
            )
            ratio_array_local = np.where(
                (ratio_array_full > 0) & (ratio_array_full < 1)
            )[0]

            if len(ratio_array_local) == 0:
                x0 = times[ratio_array_full[::-1].argmin()]
                bounds = (
                    times[ratio_array_full[::-1].argmin()],
                    times[ratio_array_full.argmax()],
                )
            elif len(ratio_array_local) == 1:
                x0 = times[ratio_array_local]
                bounds = (
                    times[ratio_array_local - 1][0],
                    times[ratio_array_local + 1][0],
                )
            elif len(ratio_array_local) == 2:
                x0 = times[ratio_array_local[0]]
                bounds = (times[ratio_array_local[0]], times[ratio_array_local[-1]])
            else:
                x0 = times[ratio_array_local[0]]
                bounds = (times[ratio_array_local[0]], times[ratio_array_local[-1]])
            vals = {"x0": np.log10(x0), "bounds": np.log10(bounds)}
            val = {**vals, **ls_params}

            try:
                self.btt = up.Time(
                    10
                    ** optimize.least_squares(
                        layer_btt,
                        args=(
                            self.thick.cm,
                            self.temp.K,
                            self.efield.V,
                            self.diff,
                            self.mob,
                            self.botC / self.topC,
                            self.thick.cm,
                        ),
                        **val,
                    ).x
                )
            except ValueError:
                pass
        return self.btt.s

    def data_imp(self):
        self.data = MatDatabase().get_df(self.material, **self.data_guide)


class Module:
    def __init__(
        self,
        layers={"boro": {}, "eva": {}, "sinx": {}},
        temp=25,
        volt=1500,
        array=100,
        focus="eva",
    ):

        self.layers = layers
        self.layer_list = [Layer(material=layer, temp=temp) for layer in layers]
        self.sys_temp = temp
        self.sys_volt = volt

        self.focus = focus

        self.array_size = array
        self.T_1D = np.linspace(20, 100, self.array_size)
        self.E_1D = np.logspace(3, 6, self.array_size)
        self.T_2D, self.E_2D = np.meshgrid(self.T_1D, self.E_1D)

        self.bbt_arr = {}
        self.bbt_df = {}

    @property
    def layer_list(self):
        """Return sum of squared errors (pred vs actual)."""
        return self._layer_list

    @layer_list.setter
    def layer_list(self, lists):
        """Return sum of squared errors (pred vs actual)."""
        self._layer_list = lists

    @property
    def module(self):
        """Return sum of squared errors (pred vs actual)."""
        return pd.DataFrame(self.layer_list, index=self.layers)

    @property
    def focus(self):
        return self._focus

    @focus.setter
    def focus(self, layer_focus="eva"):
        self._focus = layer_focus.lower()

    @property
    def focus_df(self):
        return self.module.iloc[self.layers.index(self.focus), :]

    @property
    def focus_list(self):
        return self.layer_list[self.layers.index(self.focus)]

    @property
    def sys_temp(self):
        return self._sys_temp

    @sys_temp.setter
    def sys_temp(self, temp):
        self._sys_temp = temp
        [setattr(self.layer_list[x], "temp", temp) for x in range(len(self.layer_list))]

    @property
    def sys_volt(self):
        return self._sys_volt

    @sys_volt.setter
    def sys_volt(self, volt):
        self._sys_volt = volt
        [
            setattr(
                self.layer_list[x],
                "efield",
                self.resistivity(self.module.iloc[x, 0]) * volt,
            )
            for x in range(len(self.layer_list))
        ]

    @property
    def ext_volt(self):
        return self.focus_df["efield"] / self.resistivity()

    @property
    def sys_res(self):
        return self.module["res"].sum()

    def thickness_adj(
        self,
        layer=0,
        thick=0,
        by_module=False,
        module=gf.tocm(4.55, "mm"),
        cell=gf.tocm(200, "um"),
        backsheet=gf.tocm(300, "um"),
        glass="soda",
        enc=0,
    ):
        if type(layer) is list:
            for i, lay in enumerate(layer):
                self.layer_list[lay].thick = thick[i]
        else:
            self.layer_list[layer].thick = thick
        if by_module:
            self.layer_list[enc] = (
                module - self.module.loc[glass, "thick"] - cell - backsheet
            ) / 2

    def resistivity(self, focus=None, temp=None):
        if temp is not None:
            self.set_temp(temp)
        if focus is None:
            focus = self.focus
        # should be the resistivity of the layer that is to be applied against the resistance
        return self.module.resistivity[self.layers.index(focus)] / sum(
            self.module.thick * self.module.resistivity
        )

    def bbt(
        self,
        ident,
        layer="EVA",
        target=1e16,
        source=5e21,
        diff_range="Ave",
        diff_depth=0,
        thick=None,
    ):
        if thick is None:
            thick = self.module.thick[self.layers.index(layer.lower())]
        stresses_list = [
            Layer(
                material=layer,
                diff_type=diff_range,
                temp=temps,
                efield=field,
                thick=thick,
            )
            for temps in self.T_1D
            for field in self.E_1D
        ]
        info_df = pd.DataFrame(stresses_list)

        if diff_depth == 0:
            diff_depth = thick
        info_df["depth"] = diff_depth
        ratio = target / source

        test1 = c_ratio_np_df(1, info_df)
        time_bound_low = np.full(len(test1), 0.0)
        time_bound_high = np.full(len(test1), 0.0)
        cont = 0
        changed_logic = np.full(len(test1), True)

        secs = 0
        while cont == 0:
            if secs <= 60:  # if less than a min
                delta = 1  # inc by sec
            elif secs <= 3600 * 24:  # if less than a hour
                delta = float(60)  # inc by min
            elif secs <= 3600 * 24 * 30:  # if less than a month
                delta = float(3600)  # inc by hour
            # elif secs <= 3600*24*7: # if less than a week
            #     delta=float(3600*24) # inc by day
            elif secs <= 3600 * 24 * 365 * 0.5:  # if less than half a year
                delta = float(3600 * 24)  # inc by day
            elif secs <= 3600 * 24 * 365:  # if less than a year
                delta = float(3600 * 24 * 7)  # inc by week
            elif secs <= 3600 * 24 * 365 * 100:  # if less than 100 year
                delta = float(3600 * 24 * 14)  # inc by 2 week
            elif secs > 3600 * 24 * 365 * 250:  # if less than 250 year
                delta = float(3600 * 24 * 30)  # inc by month
            elif secs > 3600 * 24 * 365 * 500:  # if less than 500 year
                delta = float(3600 * 24 * 365)  # inc by year
            elif secs > 3600 * 24 * 365 * 1000:
                delta = float(secs * 2)  # double each round
            secs += delta
            test2 = c_ratio_np_df(secs, info_df)

            time_bound_low[test1 == 0] = float(secs)

            comb_logic = (test2 - test1) <= 0.1
            test_logic = test2 >= 0.5
            changed_logic[comb_logic * test_logic] = False

            time_bound_high[((test2 - test1) > 0) * changed_logic] = float(secs)
            time_bound_low[time_bound_low >= time_bound_high] = secs - delta

            test1 = test2

            if np.max(time_bound_high) < secs and np.min(time_bound_high) != 0:
                cont = 1
        self.info = info_df
        # self.info['time'] = 1
        self.info["low"] = time_bound_low
        self.info["high"] = time_bound_high

        self.info["time"] = np.array(
            [
                optimize.minimize_scalar(
                    find_tauc_np_cratio_3d_df,
                    bounds=(self.info["low"][y], self.info["high"][y]),
                    args=(self.info.loc[y, :], ratio),
                    method="bounded",
                ).x
                for y in range(len(self.info))
            ]
        )

        self.bbt_arr[ident] = self.info.pivot_table(
            values="time", columns="temp", index="efield"
        )
        self.bbt_df[ident] = self.info

    def find_time(self, ident, temp=None, field=None, volt=None):
        if ident not in self.bbt_arr:
            print("Run Simulation first")
            return
        df = self.bbt_arr[ident]
        if temp is not None:
            self.sys_temp = df["temp"].sub(temp).abs().min()
        if volt is not None:
            self.sys_volt = df["volt"].sub(volt).abs().min()
        if field is not None:
            field = df["efield"].sub(field).abs().min()
        return df["time"][(df["temp"] == self.sys_temp) & (df["efield"] == field)]

    def leakage(self, area=None, temp=None, volt=None):

        if area is not None:
            [
                setattr(self.layer_list[x], "area", area)
                for x in range(len(self.layer_list))
            ]
        if temp is None:
            temp = self.sys_temp
        else:
            self.sys_temp = temp
        if volt is None:
            volt = self.sys_volt
        else:
            self.sys_volt = volt
        self.I_leakage = volt / self.sys_res

        return self.I_leakage

    def find_diff(self, area=None, temp=None, volt=None):

        if area is not None:
            [
                setattr(self.layer_list[x], "area", area)
                for x in range(len(self.layer_list))
            ]
        if temp is None:
            temp = self.sys_temp
        else:
            self.sys_temp = temp
        if volt is None:
            volt = self.sys_volt
        else:
            self.sys_volt = volt
        self.I_leakage = volt / self.sys_res

        return self.I_leakage


#%%  BTT general sim

mod_root = {
    "soda": {"thick": "metric"},
    "eva": {"resis": 1, "diff": "max", "thick": "thin"},
    "sin": {"resis": 1},
}

test_temp = 60
test_volt = 1000

testlayer = Layer2("soda", mod_root["soda"], temp=test_temp, efield=1e5)

#%%

mod_eva1 = Module(layers=["soda", "eva", "sinx"], temp=test_temp, volt=test_volt)
mod_eva1.thickness_adj(
    layer=[0, 1, 2], thick=[gf.tocm(3.2, "mm"), gf.tocm(450, "um"), gf.tocm(80, "nm")]
)
res_eva1 = mod_eva1.module

mod_eva2 = Module(layers=["soda", "eva_alt", "sinx"], temp=test_temp, volt=test_volt)
mod_eva2.thickness_adj(
    layer=[0, 1, 2], thick=[gf.tocm(3.2, "mm"), gf.tocm(450, "um"), gf.tocm(80, "nm")]
)
res_eva2 = mod_eva2.module

mod_poe1 = Module(layers=["soda", "poe_a", "sinx"], temp=test_temp, volt=test_volt)
mod_poe1.thickness_adj(
    layer=[0, 1, 2], thick=[gf.tocm(3.2, "mm"), gf.tocm(450, "um"), gf.tocm(80, "nm")]
)
res_poe1 = mod_poe1.module

mod_poe2 = Module(layers=["soda", "poe_b", "sinx"], temp=test_temp, volt=test_volt)
mod_poe2.thickness_adj(
    layer=[0, 1, 2], thick=[gf.tocm(3.2, "mm"), gf.tocm(450, "um"), gf.tocm(80, "nm")]
)
res_poe2 = mod_poe2.module

mod_poe3 = Module(layers=["soda", "poe_c", "sinx"], temp=test_temp, volt=test_volt)
mod_poe3.thickness_adj(
    layer=[0, 1, 2], thick=[gf.tocm(3.2, "mm"), gf.tocm(450, "um"), gf.tocm(80, "nm")]
)
res_poe3 = mod_poe3.module
