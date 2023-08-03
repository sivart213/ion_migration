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
import xarray as xr
import periodictable as pt

# import matplotlib.pyplot as plt
# from matplotlib import ticker
from scipy.special import erfc
from scipy import optimize

# from matplotlib.colors import LogNorm
# from scipy.optimize import curve_fit, fsolve, minimize
# from functools import partial
from dataclasses import field, fields, astuple, dataclass, InitVar
from copy import copy, deepcopy


def arrh_wrap(T, pre_fac, E_A):
    T = up.Temp(T, "C").K
    return gf.arrh(T, pre_fac, E_A)


def inv_sum_invs(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return 1 / sum(1 / arr)


def np_unpacker(var, df, target, scale):
    if "log" in scale.lower():
        var = 10 ** var
    try:
        df[target] = var
    except ValueError:
        df[target] = [var]
    time = df["time"].to_numpy()[0]
    depth = df["depth"].to_numpy()[0]
    thick = df["thick"].to_numpy()[0]
    diff = df["diff"].to_numpy()[0]
    temp = df["temp"].to_numpy()[0]
    efield = df["efield"].to_numpy()[0]

    return time, depth, thick, diff, temp, efield


def char_time(var, df, target="time", scale="lin"):
    time, depth, thick, diff, temp, efield = np_unpacker(var, df, target, scale)

    mob = diff / (gf.KB_EV * temp)

    return ((2 * np.sqrt(diff * time)) + mob * efield * time) - depth


def np_ratio(var, df, target="time", scale="lin"):  # thick,temp,efield,time,diff,mob):
    time, depth, thick, diff, temp, efield = np_unpacker(var, df, target, scale)

    mob = diff / (gf.KB_EV * temp)

    term_A1 = erfc((depth - mob * efield * time) / (2 * np.sqrt(diff * time)))
    term_A2 = erfc(
        -(depth - 2 * thick + mob * efield * time) / (2 * np.sqrt(diff * time))
    )
    term_B = erfc(-mob * efield * time / (2 * np.sqrt(diff * time)))
    return (1 / (2 * term_B)) * (term_A1 + term_A2)


def np_cost(var, df, ratio=0.08, target="time", scale="lin"):

    np_res = np_ratio(var, df, target, scale)
    if isinstance(np_res, float):
        if np_res < 1e-10:
            np_res = 1
    else:
        for n in range(len(np_res)):
            if np_res[n] < 1e-10:
                np_res[n] = 1
    return (np_res - ratio) ** 2


class NernstPlanck(object):
    def __init__(self, df):
        self.df = df

    def char_eq(self, target="time", xmax=1e12, bound=0.5):
        """Return sum of squared errors (pred vs actual)."""

        x0 = optimize.root_scalar(
            char_time, bracket=(0, xmax), args=(self.df.copy(), target), method="bisect"
        ).root
        bounds = (x0 * (1 - bound), x0 * (1 + bound))

        return {"x0": x0, "bounds": bounds}

    def np_sim(
        self, target="time", ratio=None, scale="lin", ls_params={}, **pre_kwargs
    ):
        """Return sum of squared errors (pred vs actual)."""

        ls_params = {**{"jac": "3-point", "xtol": 1e-12}, **ls_params}

        vals = self.char_eq("time", **pre_kwargs)

        val = {**vals, **ls_params}

        if ratio is None and "topC" in self.df.columns:
            ratio = self.df.botC / self.df.topC
        else:
            ratio = 0.08
        try:
            results = optimize.least_squares(
                np_cost, args=(self.df.copy(), ratio, target, scale,), **val,
            ).x[0]
        except ValueError:
            results = 0
        return results


class MatDatabase(object):
    def __init__(self):
        self.path = "C:\\Users\\j2cle\\Work Docs\\Data\\Databases"
        self.file = "material_data.xlsx"
        self.database = pd.read_excel(f"{self.path}\\{self.file}", index_col=[0, 1, 2])

    @property
    def material(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_material"):
            self._material = "mat"
        return self._material

    @material.setter
    def material(self, val):
        if self.database.index.isin([val], level=0).any():
            self._material = val

    @property
    def mat_data(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.database.loc[self.material, :]

    def get_df(self, material, resis=1, perm=1, diff=1, thick=1):
        self.material = material
        self.data = self.mat_data.xs(1, level="label").copy()
        labels = {"resis": resis, "perm": perm, "diff": diff, "thick": thick}

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


class Layer(object):
    """Return sum of squared errors (pred vs actual)."""

    def __init__(
        self,
        material="mat",
        data_guide={"resis": 1, "perm": 1, "diff": 1, "thick": 1},
        temp=25,
        efield=0,
        volt=0,
        area=1,
        topC=1e21,
        botC=1e10,
        btt_solver=False,
        rt_link=True,
        dt_link=False,
    ):
        """Return layer object."""
        # mat_test1=MatDatabase()
        # Invisible attribute (init-only)

        self.material = material
        self.data_guide = data_guide

        self.data_imp()

        # Initialized attribute
        self.temp = temp

        if efield == 0 and volt != 0:
            self.volt = volt
        else:
            self.efield = efield
        # Generated attribute
        self.area = area

        self.topC = topC
        self.botC = botC

        self.time = up.Time(1)

        self.lay_np = NernstPlanck(self.np_info)

        self.rt_link = rt_link
        self.dt_link = dt_link

    def __getitem__(self, item):
        """Return sum of squared errors (pred vs actual)."""
        return getattr(self, item)

    def __setitem__(self, item, val):
        """Return sum of squared errors (pred vs actual)."""
        setattr(self, item, val)

    @property
    def thick(self):
        """Return thickness of layer."""
        return up.Length(self.data.var1["thick"], self.data.var2["thick"], 1, "cm")

    @thick.setter
    def thick(self, val):
        if val is not None:
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
                self.data.var1["thick"] = val
                self.data.var2["thick"] = "cm"

    @property
    def temp(self):
        """Return thickness of layer."""
        return self._temp

    @temp.setter
    def temp(self, val):
        if val is not None:
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
        """ Return Area """
        return self._area

    @area.setter
    def area(self, val):
        if val is not None:
            if isinstance(val, up.Length):
                self._area = val
            elif isinstance(val, (tuple, list)):
                self._area = up.Length(*val)
            elif isinstance(val, (dict)):
                self._area = up.Length(**val)
            else:
                self._area = up.Length(val, "cm", 2)
        if self._area.exp != 2:
            self._area = up.Length(self._area.value ** 2, self._area.unit, 2, "cm")

    @property
    def efield(self):
        """ Return V/cm as volt type asuming cm's """
        return self._efield

    @efield.setter
    def efield(self, val):
        """Ensures input is a Volt class"""
        if val is not None:
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
        """ rho = rho0 * exp(Ea / k_B T) """
        return self.data.var1["resis"] * np.exp(
            -1*self.data.var2["resis"] / (gf.KB_EV * self.temp.K)
        )

    @resistivity.setter
    def resistivity(self, val):
        """ sets rho_0 to val """
        if val is not None:
            if isinstance(val, up.Res):
                val = val
            elif isinstance(val, (tuple, list)):
                val = up.Res(*val)
            elif isinstance(val, (dict)):
                val = up.Res(**val)
            else:
                val = up.Res(val, "Ohm")

            if self.rt_link and self.data.var2["resis"] != 0:
                self.temp = up.Temp(
                    -1*self.data.var2["resis"]
                    / (gf.KB_EV * np.log(val.Ohm / self.data.var1["resis"])),
                    "K",
                )
            else:
                self.data.var1["resis"] = val.Ohm
                self.data.var2["resis"] = 0

    @property
    def volt(self):
        """ V = E * t """
        return up.Volt(self.efield.V * self.thick.cm)

    @volt.setter
    def volt(self, val):
        """ E = V / t """
        if val is not None:
            self.efield = val / self.thick.cm

    @property
    def resistance(self):
        """ R = rho * t / A """
        return up.Res(self.resistivity * self.thick.cm / self.area.cm)

    @resistance.setter
    def resistance(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            if isinstance(val, up.BaseUnits):
                self.resistivity = val.Ohm * self.area.cm / self.thick.cm
            else:
                self.resistivity = val * self.area.cm / self.thick.cm

    @property
    def curr(self):
        """Return sum of squared errors (pred vs actual)."""
        return up.Curr(self.volt.V / self.resistance.Ohm)

    @curr.setter
    def curr(self, val):
        """ E = I * R / t """
        if val is not None:
            self.efield = val * self.resistance.Ohm / self.thick.cm

    @property
    def er(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_er"):
            self._er = self.data.var1["perm"]
        return self._er

    @er.setter
    def er(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            val = gf.Sig_figs(val, 2)
            if val <= 1:
                self._er = 1
            else:
                self._er = val

    @property
    def cap(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.er * up.Length(gf.PERM, "m", -1).cm * self.area.cm / self.thick.cm

    @cap.setter
    def cap(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            self.er = (
                val * self.thick.cm / (self.area.cm * up.Length(gf.PERM, "m", -1).cm)
            )

    @property
    def charge(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.er * up.Length(gf.PERM, "m", -1).cm * self.resistivity * self.curr.A

    @charge.setter
    def charge(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            self.er = val / (
                up.Length(gf.PERM, "m", -1).cm * self.resistivity * self.curr.A
            )

    @property
    def time(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_time"):
            self._time = up.Time(24, "hr", 1, "s")
        return self._time

    @time.setter
    def time(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            if isinstance(val, up.Time):
                self._time = val
            elif isinstance(val, (tuple, list)):
                self._time = up.Time(*val)
            elif isinstance(val, (dict)):
                self._time = up.Time(**val)
            else:
                self._time = up.Time(val, "s")

    @property
    def depth(self):
        """Return thickness of layer."""
        if not hasattr(self, "_depth"):
            self._depth = self.thick
        return self._depth

    @depth.setter
    def depth(self, val):
        if val is not None:
            if isinstance(val, up.Length):
                self._depth = val
            elif isinstance(val, (tuple, list)):
                self._depth = up.Length(*val)
            elif isinstance(val, (dict)):
                self._depth = up.Length(**val)
            else:
                self._depth = up.Length(val, "cm")

    @property
    def diff(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.data.var1["diff"] * np.exp(
            -1*self.data.var2["diff"] / (gf.KB_EV * self.temp.K)
        )

    @diff.setter
    def diff(self, val):
        if val is not None:
            if self.dt_link and self.data.var2["diff"] != 0:
                self.temp = up.Temp(
                    -1 * self.data.var2["diff"]
                    / (gf.KB_EV * np.log(val / self.data.var1["diff"])),
                    "K",
                )
            else:
                self.data.var1["diff"] = val
                self.data.var2["diff"] = 0

    @property
    def btt(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_btt"):
            self.depth = self.thick
            self._btt = self.np_solver("time")
            self.time = self._btt
        return up.Time(self._btt, "s")

    @btt.setter
    def btt(self, val):
        """Return sum of squared errors (pred vs actual)."""
        self.time = val
        if hasattr(self, "_btt"):
            del self._btt

    @property
    def info(self):
        """Return the necessary information."""

        info = {
            "mat": str(self.material),
            "thickness (um)": float(self.thick.um),
            "temperature (C)": float(self.temp.C),
            "efield (MV/cm)": float(self.efield.MV),
            "resistivity (ohm.cm)": float(self.resistivity),
            "volt (V)": float(self.volt.V),
            "resistance (ohm)": float(self.resistance.Ohm),
            "curr (A)": float(self.curr.A),
            "er": float(self.er),
            "capacitance (F)": float(self.cap),
            "charge": float(self.charge),
        }

        return pd.DataFrame([info])

    @property
    def np_info(self):
        """Return the necessary information."""

        df = {
            "time": float(self.time.s),
            "depth": float(self.depth.cm),
            "thick": float(self.thick.cm),
            "diff": float(self.diff),
            "temp": float(self.temp.K),
            "efield": float(self.efield.V),
            "top conc": float(self.topC),
            "bot conc": float(self.botC),
        }

        return pd.DataFrame([df])

    def np_solver(
        self, target, topC=None, botC=None, scale="lin", ls_params={}, **pre_kwargs
    ):
        """Return sum of squared errors (pred vs actual)."""
        if self.diff != 0 or target == "diff":
            if topC is not None:
                self.topC = topC
            if botC is not None:
                self.botC = botC
            ratio = self.botC / self.topC
            np = NernstPlanck(self.np_info)
            return np.np_sim(target, ratio, scale, ls_params, **pre_kwargs)
        else:
            return 0

    def data_imp(self, material=None, data_guide=None):
        if material is not None:
            self.material = material
        if data_guide is not None:
            self.data_guide = data_guide
        self.data = MatDatabase().get_df(self.material, **self.data_guide)


# @dataclass
# class obj_vect(object):
#     obj: list = []

#     def __getitem__(self, item):
#         """Return sum of squared errors (pred vs actual)."""
#         return getattr(self, item)


class Stack:
    def __init__(
        self, layers=[["boro", {}], ["eva", {}], ["sinx", {}]], **layer_kwargs,
    ):

        if isinstance(layers, (int, np.integer)):
            self.layers = [Layer() for _ in range(layers)]
        elif all(isinstance(lay, Layer) for lay in layers):
            self.layers = layers
        else:
            self.layers = [
                Layer(material=mat, data_guide=info, **layer_kwargs)
                for mat, info in layers
            ]

    def __iter__(self):
        """Return sum of squared errors (pred vs actual)."""
        return iter(astuple(self))

    def __getitem__(self, item):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(item, (int, slice, np.integer)):
            return self.layers[item]
        if hasattr(self, item):
            return getattr(self, item)
        items = [getattr(lay, item) for lay in self.layers]
        if isinstance(items[0], up.BaseUnits):
            item_type = type(items[0])
            base = items[0].base_unit
            exp = items[0].exp
            print_unit = items[0].print_unit
            array = np.array([lay[base] for lay in items])
            if np.all(array == array[0]):
                array = array[0]
            return item_type(array, base, exp, print_unit)
        else:
            return np.array(items)

    def __setitem__(self, item, val):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(val, Layer):
            if isinstance(item, int):
                self.layers[item] = val
        elif hasattr(Layer, item):
            if isinstance(val, (int, float, np.integer, np.float)) or len(val) != len(
                self.layers
            ):
                [setattr(lay, item, val) for lay in self.layers]
            else:
                [setattr(self.layers[n], item, val[n]) for n in range(len(self.layers))]
        elif hasattr(self, item):
            setattr(self, item, val)

    @property
    def uniform_vars(self):
        return ["temp", "curr", "area"]

    @property
    def stack_info(self):
        """Return sum of squared errors (pred vs actual)."""
        self._stack_info = pd.concat(
            [lay.info for lay in self.layers], axis=0, ignore_index=True
        )
        return self._stack_info

    @property
    def np_stack(self):
        """Return sum of squared errors (pred vs actual)."""
        self._np_stack = pd.concat(
            [lay.np_info for lay in self.layers], axis=0, ignore_index=True
        )
        return self._np_stack


class VoltDev(Stack):
    def __init__(self, layers, sys_volt=0, **layer_kwargs):

        if isinstance(layers, Stack):
            self.layers = layers.layers
        elif isinstance(layers, (int, np.integer)):
            self.layers = [Layer() for _ in range(layers)]
        elif all(isinstance(lay, Layer) for lay in layers):
            self.layers = layers
        else:
            self.layers = [
                Layer(material=mat, data_guide=info, **layer_kwargs)
                for mat, info in layers
            ]
        self.sys_volt = sys_volt

    @property
    def sys_res(self):
        # return up.Res(self.stack.resistance.Ohm.sum())
        return up.Res(self["resistance"].Ohm.sum())

    @property
    def sys_volt(self):
        if not hasattr(self, "_sys_volt"):
            self._sys_volt = up.Volt(1)
        return self._sys_volt

    @sys_volt.setter
    def sys_volt(self, val):
        if isinstance(val, up.Volt):
            self._sys_volt = val
        elif isinstance(val, (tuple, list)):
            self._sys_volt = up.Volt(*val)
        elif isinstance(val, (dict)):
            self._sys_volt = up.Volt(**val)
        else:
            self._sys_volt = up.Volt(val, "V")
        self["volt"] = self["resistance"].Ohm / self.sys_res.Ohm * self._sys_volt.V

    @property
    def sys_cap(self):
        """Return sum of squared errors (pred vs actual)."""
        return inv_sum_invs(self["cap"])

    @property
    def sys_charge(self):
        """Return sum of squared errors (pred vs actual)."""
        # if not hasattr(self, 'free_charge') or self.free_charge is None:
        return self.sys_cap * self.sys_volt.V
        # else:
        #     return self.free_charge * self.area.cm

    @property
    def ave_charge(self):
        """Return sum of squared errors (pred vs actual)."""
        return (sum(self["charge"]) + self.sys_charge) / (len(self.layers) + 1)

    # # def bulk_param(self, params):
    # #     """Update all layer information from dict"""
    # #     try:
    # #         [[self.single_param(n, key, val[n]) for n in range(len(self.stack.layers))] for key, val in params.items()]
    # #         # [[setattr(self.stackl.layers[n], key, val[n]) for n in range(len(self.stack.layers))] for key, val in params.items()]
    # #     except (AttributeError, IndexError):
    # #         print("Format Error: Must be in form {param:[x1,x2,...,xn],...}, where n is is length of list and uniform")

    def single_param(self, layer=0, param="volt", val=0, unit=None, lock_ext=False):
        """Update single layer information  and find needed system volt to accomodate this"""
        ext = self.sys_volt.V
        r_tot = self.sys_res.Ohm
        if isinstance(unit, str):
            val = [val, unit]
        if hasattr(self, param) or param in self.uniform_vars:
            self[param] = val
        else:
            self[layer][param] = val
            if r_tot != self.sys_res.Ohm or lock_ext:
                self.sys_volt = up.Volt(ext,'V')
            else:
                self.sys_volt = (self[layer]["volt"].V * self.sys_res.Ohm / self[layer]["resistance"].Ohm)
        return

    def curr_eval(self, current=0, lay_params={}, alter_volt=False):
        """Calculate needed R to explain current and related eff_volt"""
        if isinstance(current, up.Curr):
            tar_curr = current
        else:
            tar_curr = up.Curr(current)
        if lay_params == {}:
            lay_params = {"material": "air", "temp": self["temp"], "area": self["area"]}
        added_layer = Layer(**lay_params)
        added_layer.volt = tar_curr.A * added_layer.resistance.Ohm

        eff_volt = up.Volt(self.sys_volt.V - added_layer.volt.V)

        if alter_volt:
            self.sys_volt = eff_volt
        return {"eff_volt": eff_volt, "layer": added_layer}

    def cap_eval(
        self, dampener=0.05, iters=100, qtol=0.05,
    ):
        n = 0
        dev = 1
        while dev > qtol and n < iters:
            eps0 = np.array(self["er"])
            # eps1 = (self.sys_cap * self.sys_res.Ohm) / (up.Length(gf.PERM, "m", -1).cm * self.stack.resistivity.Ohm)
            eps1 = self.sys_charge / (
                up.Length(gf.PERM, "m", -1).cm * self["efield"].V * self["area"].cm
            )
            eps_dev = eps0 - eps1
            epsf = eps0 - eps_dev * dampener

            self[abs(eps_dev / eps0).argmax()]["er"] = epsf[
                abs(eps_dev / eps0).argmax()
            ]
            # setattr(self.stack.layers[abs(eps_dev/eps0).argmax()], 'er', epsf[abs(eps_dev/eps0).argmax()])
            dev = max(abs(self.ave_charge - np.array(self["charge"])) / self.ave_charge)
            n += 1
        return


class Module(object):
    def __init__(
        self,
        stack,
        sys_volt=0,
        size=100,
        layer=1,
        dep_var="btt",
        dep_unit="value",
        ind_var={},
        **kwargs,
    ):

        if not isinstance(stack, VoltDev):
            stack = VoltDev(stack, sys_volt)
        self.stack = stack
        self.sys_volt = sys_volt
        self.size = size

        self.layer = layer
        self.dep_var = dep_var
        self.dep_unit = dep_unit

        if ind_var == {}:
            ind_var = {
                "x": {"param": "temp", "layer": 1, "unit": "C", "array": [25, 100]},
                "y": {"param": "efield", "layer": 1, "unit": "V", "array": [1e3, 1e6]},
            }
        self.ind_vars = ind_var
        self.solver(**self.ind_vars["x"])
        if "y" in self.ind_vars.keys():
            self.solver(**self.ind_vars["y"])

    @property
    def ind_vars(self):
        """ log scales: efield """
        return self._ind_vars

    @ind_vars.setter
    def ind_vars(self, val):
        for key, vals in val.items():
            if len(vals["array"]) != self.size:
                val[key]["array"] = self.range_gen(
                    min(vals["array"]), max(vals["array"]), vals["unit"]
                )
        if "y" in val.keys():
            val["x"]["array"], val["y"]["array"] = np.meshgrid(
                val["x"]["array"], val["y"]["array"]
            )
        else:
            val["x"]["array"] = [val["x"]["array"]]
        self._ind_vars = val

    @property
    def shape(self):
        """ log scales: efield """
        if "y" in self.ind_vars.keys():
            return [self.size, self.size]
        else:
            return [self.size, 1]

    @property
    def obj_matrix(self):
        """ log scales: efield """
        if not hasattr(self, "_obj_matrix"):
            self._obj_matrix = [
                [deepcopy(self.stack) for _ in range(self.shape[0])]
                for _ in range(self.shape[1])
            ]
        return self._obj_matrix

    def result(self, layer=None, dep_var=None, dep_unit=None):
        """ log scales: efield """

        if layer is None:
            layer = self.layer
        if dep_var is None:
            dep_var = self.dep_var
        if dep_unit is None:
            dep_unit = self.dep_unit
        unitless = ["resistivity", "diff", "er", "cap", "charge"]

        if dep_var in unitless:
            if len(self.ind_vars) == 1:
                res_matrix = [obj[layer][dep_var] for obj in self.obj_matrix[0]]
                res_matrix = xr.DataArray(np.array(res_matrix), coords=self.ind_vars['x']['array'])
            else:
                res_matrix = [
                    [self.obj_matrix[m][n][layer][dep_var] for n in range(self.size)]
                    for m in range(self.size)
                ]
                res_matrix = xr.DataArray(np.array(res_matrix), coords=[self.ind_vars['x']['array'][0,:],self.ind_vars['y']['array'][:,0]])
        else:
            if len(self.ind_vars) == 1:
                res_matrix = [obj[layer][dep_var][dep_unit] for obj in self.obj_matrix[0]]
                res_matrix = xr.DataArray(np.array(res_matrix), coords=self.ind_vars['x']['array'])
            else:
                res_matrix = [
                    [
                        self.obj_matrix[m][n][layer][dep_var][dep_unit]
                        for n in range(self.size)
                    ]
                    for m in range(self.size)
                ]
                res_matrix = xr.DataArray(np.array(res_matrix), coords=[self.ind_vars['x']['array'][0,:],self.ind_vars['y']['array'][:,0]])
        return res_matrix

    def range_gen(self, min_val, max_val, unit, scale=None):

        if scale is None:
            if np.log10(max_val / min_val) > 1:
                scale = "log"
            else:
                scale = "lin"
        if scale == "log":
            array = np.logspace(np.log10(min_val), np.log10(max_val), self.size)
        else:
            array = np.linspace(min_val, max_val, self.size)
        return array

    def solver(self, param, unit, array, layer=None):
        if layer is None:
            layer = self.layer
        objs = self.obj_matrix

        if param != "btt":
            [
                [
                    objs[m][n].single_param(layer, param, [array[m][n], unit])
                    for n in range(self.shape[0])
                ]
                for m in range(self.shape[1])
            ]
        else:
            for n in range(self.shape[0]):
                for m in range(self.shape[1]):
                    objs[m][n][layer]["btt"] = array[n]
                    objs[m][n][layer].np_solver(self.dep_var)
        return objs


#%% Init Prams

init_slg_params = [
    ["soda", {"resis": 2, "thick": "metric"}],
    ["eva", {"resis": 1, "diff": "max", "thick": "common"}],
    ["sinx", {"resis": 4, "diff":2}],
]

vidr_slg_params = [
    ["soda", {"resis": 1, "thick": "metric"}],
    ["eva", {"resis": 1, "diff": "max", "thick": "common"}],
    ["sinx", {"resis": 4, "diff":2}],
]

ti_bar_slg_params = [
    ["soda", {"resis": 1, "thick": "metric"}],
    ["tio2", {"resis": 1}],
    ["eva", {"resis": 1, "diff": "max", "thick": "common"}],
    ["sinx", {"resis": 1, "diff":2}],
]


#%% Init Vars

temp_var = {"param": "temp", "layer": 1, "unit": "C", "array": [25, 100]}

resist_var = {'param':'resistivity','layer':1,'unit':'Ohm','array':[1e12, 1e16]}

efield_var = {"param": "efield", "layer": 1, "unit": "V", "array": [1e3, 1e6]}

sys_volt_var = {"param": "sys_volt", "layer": 1, "unit": "V", "array": [10, 2000]}

thick_var = {"param": "thick", "layer": 1, "unit": "um", "array": [100, 600]}

btt_var = { "x": temp_var, "y": efield_var}

#%% Sims

init_slg = Module(init_slg_params, 1500, 5, 1, "btt", "s", btt_var)

# btt_init_slg = init_slg.result().to_dataframe('value').unstack("dim_0")
# resis_init_slg = init_slg.result(1, 'resistivity').to_dataframe('value').unstack("dim_0")
# efield_init_slg = init_slg.result(2, 'efield','V').to_dataframe('value').unstack("dim_0")

# vidr_slg = Module(vidr_slg_params, 1500, 25, 1, "btt", "s", btt_var)
# btt_vidr_slg = vidr_slg.result()
# # resis_vidr_slg = vidr_slg.result(1, 'resistivity')
# efield_vidr_slg = vidr_slg.result(2, 'efield','V')

# barrier_slg = Module(ti_bar_slg_params, 1500, 25, 1, "btt", "s", btt_var)
# btt_barrier_slg = barrier_slg.result()
# # resis_vidr_slg = barrier_slg.result(1, 'resistivity')
# efield_barrier_slg = barrier_slg.result(2, 'efield','V')

# lay_init = [Layer(material=mat, data_guide=info, temp=60, area=1, topC=1e19, botC=1e16) for mat, info in vidr_slg_params]

# vidr_slg_resis = Module(lay_init, 1500, 50, 1, "btt", "s", {'x':resist_var})

# efield_vidr_slg_resis = vidr_slg_resis.result(2, 'efield','V').to_dataframe('value')
# btt_vidr_slg_resis = vidr_slg_resis.result(2, 'btt', 's').to_dataframe('value')

# print_1d = btt_vidr_slg_resis.to_dataframe('value')
# print_2d = btt_init_slg.to_dataframe('value').unstack("dim_0")


# #%%
# time = up.Time(np.linspace(60,3600*24*5),'s',1,'d')
# df = vidr_slg_resis.obj_matrix[0][0][2]['np_info']
# df['efield'] = 0
# df['diff'] = 2e-17
# res = np_ratio(time.s,df)
