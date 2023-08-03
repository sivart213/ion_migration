# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:42:38 2021

@author: j2cle
"""
# %%
import numpy as np
import pandas as pd
import General_functions as gf
import re
import General_eqns as ge
import Units_Primary4 as up
import xarray as xr
import periodictable as pt
import sympy as sym
from itertools import product

# import matplotlib.pyplot as plt
# from matplotlib import ticker
from scipy.special import erfc
from scipy import optimize

# from matplotlib.colors import LogNorm
# from scipy.optimize import curve_fit, fsolve, minimize
# from functools import partial
from dataclasses import field, fields, astuple, dataclass, InitVar
from copy import copy, deepcopy


# def xarr_to_dict


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
    dif = df["dif"].to_numpy()[0]
    temp = df["temp"].to_numpy()[0]
    efield = df["efield"].to_numpy()[0]

    return time, depth, thick, dif, temp, efield


def char_time(var, df, target="time", scale="lin"):
    time, depth, thick, dif, temp, efield = np_unpacker(var, df, target, scale)

    mob = dif / (gf.KB_EV * temp)

    return ((2 * np.sqrt(dif * time)) + mob * efield * time) - depth


def np_ratio(var, df, target="time", scale="lin"):  # thick,temp,efield,time,dif,mob):
    time, depth, thick, dif, temp, efield = np_unpacker(var, df, target, scale)

    mob = dif / (gf.KB_EV * temp)

    term_A1 = erfc((depth - mob * efield * time) / (2 * np.sqrt(dif * time)))
    term_A2 = erfc(
        -(depth - 2 * thick + mob * efield * time) / (2 * np.sqrt(dif * time))
    )
    term_B = erfc(-mob * efield * time / (2 * np.sqrt(dif * time)))
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


def eqn_sets(params, **kwargs):

    results = []
    for key, vals in params.items():
        if isinstance(vals, (np.ndarray, list, tuple)):
            tmp = [eqn({**params, **{key: val}}, **kwargs) for val in vals]
            results.append(tmp)
    if results == []:
        results = eqn(params, **kwargs)
    return results


def eqn(params, target="D", eqns="x-1", as_set=True):

    x = sym.Symbol("x", positive=True)
    params[target] = x

    if not isinstance(eqns, str):
        expr = sym.parsing.sympy_parser.parse_expr(eqns.pop(0)[1])
        expr = expr.subs(eqns)
    else:
        expr = sym.parsing.sympy_parser.parse_expr(eqns)
    expr = expr.subs(params)

    try:
        res = sym.solveset(expr, x, domain=sym.S.Reals)
        if isinstance(res, sym.sets.sets.EmptySet):
            res = sym.FiniteSet(*sym.solve(expr))
    except Exception:
        res = sym.FiniteSet(*sym.solve(expr))
    if not as_set:
        res = float(list(res)[0])
    return res


class NernstPlanck(object):
    def __init__(self, df):
        self.df = df

    def char_eq(self, target="time", as_set=False):
        """Return sum of squared errors (pred vs actual)."""

        param_values = self.df.loc[0, :].to_dict()
        param_values["k_b"] = gf.KB_EV

        char = "2*(dif*time)**(1/2)+dif/(k_b*temp)*efield*time-thick"

        x0 = eqn_sets(param_values, target=target, eqns=char, as_set=False)

        return x0

    def np_sim(
        self,
        target="time",
        ratio=None,
        scale="lin",
        ls_params={},
        bound=0.5,
        **pre_kwargs,
    ):
        """Return sum of squared errors (pred vs actual)."""

        ls_params = {**{"jac": "3-point", "xtol": 1e-12}, **ls_params}

        x0 = self.char_eq(target, **pre_kwargs)
        bounds = (x0 * (1 - bound), x0 * (1 + bound))

        val = {**{"x0": x0, "bounds": bounds}, **ls_params}

        if ratio is None and "sourceC" in self.df.columns:
            ratio = self.df.conc / self.df.sourceC
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

    def get_df(self, material, resis=1, perm=1, dif=1, thick=1):
        self.material = material
        self.data = self.mat_data.xs(1, level="label").copy()
        labels = {"resis": resis, "perm": perm, "dif": dif, "thick": thick}

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
        data_guide={"resis": 1, "perm": 1, "dif": 1, "thick": 1},
        temp=25,
        efield=0,
        volt=0,
        area=1,
        sourceC=1e21,
        conc=1e10,
        btt_solver="char_time",
        rt_link=False,
        dt_link=False,
        # **kwargs,
    ):
        """Return layer object."""
        # mat_test1=MatDatabase()
        # Invisible attribute (init-only)

        self.material = material
        self.data_guide = data_guide

        self.data_imp()

        # Initialized attribute

        if efield == 0 and volt != 0:
            self.volt = volt
        else:
            self.efield = efield
        # Generated attribute
        self.temp = temp
        self.area = area

        self.sourceC = sourceC
        self.conc = conc

        self.btt_solver = btt_solver

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
        if hasattr(self, "_btt"):
            del self._btt
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
        if hasattr(self, "_btt"):
            del self._btt
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
        if hasattr(self, "_btt"):
            del self._btt
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
            -1 * self.data.var2["resis"] / (gf.KB_EV * self.temp.K)
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
                    -1
                    * self.data.var2["resis"]
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
    def time(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_time"):
            if hasattr(self, "_btt"):
                self._time = self.btt
            else:
                self._time = up.Time(4, "d", 1, "s")
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
    def dif(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.data.var1["dif"] * np.exp(
            -1 * self.data.var2["dif"] / (gf.KB_EV * self.temp.K)
        )

    @dif.setter
    def dif(self, val):
        if hasattr(self, "_btt"):
            del self._btt
        if isinstance(val, list):
            val = val[0]
        if val is not None:
            if self.dt_link and self.data.var2["dif"] != 0:
                self.temp = up.Temp(
                    -1
                    * self.data.var2["dif"]
                    / (gf.KB_EV * np.log(val / self.data.var1["dif"])),
                    "K",
                )
            else:
                self.data.var1["dif"] = val
                self.data.var2["dif"] = 0

    @property
    def btt(self):
        """Calculate btt, sets time to btt"""
        if not hasattr(self, "_btt"):
            dep = self.depth.cm
            self.depth = self.thick
            self._btt = up.Time(self.np_solver("time"), "s")
            self.depth = up.Length(dep, "cm")
        return self._btt

    @btt.setter
    def btt(self, val):
        """Calculate btt, sets time to btt"""
        if hasattr(self, "_btt"):
            del self._btt

    @property
    def info(self):
        """Return the necessary information."""
        info = {
            "mat": str(self.material),
            "area (cm2)": float(self.area.cm),
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
            "dif": float(self.dif),
            "temp": float(self.temp.K),
            "efield": float(self.efield.V),
            "sourceC": float(self.sourceC),
            "conc": float(self.conc),
        }
        return pd.DataFrame([df])

    @property
    def layer_attr(self):
        """Return the necessary information."""
        info = {
            "material": str(self.material),
            "thick": (self.thick),
            "temp": (self.temp),
            "efield": (self.efield),
            "resistivity": (self.resistivity),
            "volt": (self.volt),
            "resistance": (self.resistance),
            "curr": (self.curr),
            "er": (self.er),
            "cap": (self.cap),
            "charge": (self.charge),
            "sourceC": (self.sourceC),
            "conc": (self.conc),
            "dif": (self.dif),
            "time": (self.time),
            "btt": (self.btt),
            "btt_solver": (self.btt_solver),
        }
        return pd.DataFrame([info])

    def np_solver(
        self, target, sourceC=None, conc=None, scale="lin", ls_params={}, **pre_kwargs
    ):
        """Return sum of squared errors (pred vs actual)."""
        if self.dif != 0 and target in self.np_info.columns:
            if sourceC is not None:
                self.sourceC = sourceC
            if conc is not None:
                self.conc = conc
            ratio = self.conc / self.sourceC
            nernst = NernstPlanck(self.np_info)
            if "np" in self.btt_solver:
                return nernst.np_sim(target, ratio, scale, ls_params, **pre_kwargs)
            else:
                return nernst.char_eq(target, **pre_kwargs)
        else:
            return 1

    def data_imp(self, material=None, data_guide=None):
        if material is not None:
            self.material = material
        if data_guide is not None:
            self.data_guide = data_guide
        self.data = MatDatabase().get_df(self.material, **self.data_guide)


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
        df = pd.concat([lay.info for lay in self.layers], axis=0, ignore_index=True)
        return df

    @property
    def stack_np(self):
        """Return sum of squared errors (pred vs actual)."""
        df = pd.concat([lay.np_info for lay in self.layers], axis=0, ignore_index=True)
        return df

    @property
    def stack_data(self):
        """Return sum of squared errors (pred vs actual)."""
        df = pd.concat(
            [lay.data.loc[:, "note"] for lay in self.layers], axis=1, ignore_index=True
        ).T

        return df


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

    @property
    def sys_info(self):
        """Return the necessary information."""
        df = {
            "sys_res": self.sys_res,
            "sys_res": self.sys_res,
            "sys_volt": self.sys_volt,
            "sys_cap": self.sys_cap,
            "sys_charge": self.sys_charge,
            "ave_charge": self.ave_charge,
        }
        return pd.DataFrame([df])

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
                self.sys_volt = up.Volt(ext, "V")
            else:
                self.sys_volt = (
                    self[layer]["volt"].V
                    * self.sys_res.Ohm
                    / self[layer]["resistance"].Ohm
                )
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


# class DevMesh(object):
#     def __init__(
#         self,
#         stack,
#         thick_max,
#         thick_min,
#         curvature="plane",
#         size=100,
#         sys_volt=0,
#         **kwargs,
#     ):

#         if not isinstance(stack, VoltDev):
#             stack = VoltDev(stack, sys_volt, **kwargs)
#         self.stack = stack
#         self.sys_volt = sys_volt
#         self.size = size


#     def coordinates(self):
#         leng = np.sqrt(self.stack.area.cm) / 2

#         leng = np.sqrt(thin_stack["area"].cm) / 2
#         leng = np.linspace(-leng, leng, 100)
#         x, y = np.meshgrid(leng, np.flip(leng))
#         x_coords = leng
#         y_coords = np.flip(leng)

#         cyl_r = ge.arc(h=up.Length(10,"um").cm, a=leng[0])
#         cyl_sub = cyl_r - up.Length(10,"um").cm
#         cyl_h = np.array([[np.sqrt(cyl_r**2 - n**2) - cyl_sub for n in leng] for m in np.flip(leng)])


#         shere_r = ge.arc(h=up.Length(10,"um").cm, a=[x[0, 0], y[0, 0]])
#         sphere_sub = shere_r - up.Length(10,"um").cm
#         sphere_h = np.array([[np.sqrt(shere_r**2 - (n**2 + m**2)) - sphere_sub for n in leng] for m in np.flip(leng)])


#         return


class Module(object):
    def __init__(
        self, stack, sys_volt=0, size=100, ind_var=[], dep_var=[], **kwargs,
    ):

        if not isinstance(stack, VoltDev):
            stack = VoltDev(stack, sys_volt, **kwargs)
        self.stack = stack
        self.sys_volt = sys_volt
        self.size = size

        self.path = "C:\\Users\\j2cle\\Work Docs\\Python Scripts\\Active\\Database"
        self.folder = "MigrationModules"

        if dep_var == []:
            dep_var = [
                {"param": "btt", "layer": 1, "unit": "s"},
            ]
        self.dep_vars = dep_var

        if ind_var == []:
            ind_var = [
                {"param": "temp", "layer": 1, "unit": "C", "array": [25, 100]},
                {"param": "efield", "layer": 1, "unit": "V", "array": [1e3, 1e6]},
            ]
        self.ind_vars = ind_var

        self.stacks
        # self.solver(**self.ind_vars.loc[0, :].to_dict())
        # if "y" in self.ind_vars.keys():
        #     self.solver(**self.ind_vars["y"])

    def __getitem__(self, item):
        """Return sum of squared errors (pred vs actual)."""
        return getattr(self, item)

    @property
    def ind_vars(self):
        """ log scales: efield """
        return self._ind_vars

    @ind_vars.setter
    def ind_vars(self, val):
        if not isinstance(val, list):
            val = pd.DataFrame([val])
        else:
            val = pd.DataFrame(val)
        val["array"] = val["array"].map(self.range_gen)

        self._arrays = pd.DataFrame(list(product(*val["array"])), columns=val["param"])
        self._ind_vars = val

    @property
    def dep_vars(self):
        """ log scales: efield """
        return self._dep_vars

    @dep_vars.setter
    def dep_vars(self, val):
        if not isinstance(val, list):
            self._dep_vars = pd.DataFrame([val])
        else:
            self._dep_vars = pd.DataFrame(val)

    @property
    def arrays(self):
        """ log scales: efield """
        return self._arrays

    @property
    def sim_name(self):
        """ log scales: efield """
        if not hasattr(self, "_sim_name"):
            x = "_".join(mat for mat in self.stack.stack_info["mat"])
            y = "_".join(par for par in self.ind_vars["param"])
            self._sim_name = "_".join((x, y))
        return self._sim_name

    @sim_name.setter
    def sim_name(self, name):
        self._sim_name = name

    # @property
    # def matrix(self):
    #     """ log scales: efield """
    #     if not hasattr(self, '_matrix'):
    #         self.pickle_check()
    #     if not hasattr(self, '_matrix'):
    #         self._matrix = self.solver()
    #         self._matrix = self.conv_to_num(self.ind_vars, self._matrix)
    #         jar = gf.PickleJar(path=f"{self.path}\{self.folder}", history=True)
    #         jar[self.sim_info] = self
    #     return self._matrix

    @property
    def stacks(self):
        """ log scales: efield """
        # if not hasattr(self, '_stacks'):
        # name = self.pickle_check()
        # self.pickle_load(name)
        if not hasattr(self, "_stacks"):
            self.solver()
            # self.pickle_save()
        return self._stacks

    def pickle_check(self):
        jar = gf.PickleJar(path=f"{self.path}\{self.folder}", history=True)
        db = jar.queary(self.sim_name)
        items = ["stack_info", "stack_np", "stack_data"]
        for pickle in db:
            mod = deepcopy(jar[pickle])
            try:
                if all(
                    self.stack[i].equals(mod.stack[i]) for i in items
                ) and self.ind_vars.equals(mod.ind_vars):
                    return mod.sim_name
            except NameError:
                break
        return "error"

    def pickle_save(self, history=True):
        jar = gf.PickleJar(path=f"{self.path}\{self.folder}", history=history)
        jar[self.sim_name] = self

    def pickle_load(self, name, history=True):
        jar = gf.PickleJar(path=f"{self.path}\{self.folder}", history=history)
        try:
            mod = deepcopy(jar[name])
            [setattr(self, att, getattr(mod, att)) for att in vars(mod)]
        except IndexError:
            return

    def range_gen(self, min_max, scale=None):
        min_val, max_val = min_max
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

    def solver(self):
        matrix = [
            pd.DataFrame(columns=lay.layer_attr.columns) for lay in self.stack.layers
        ]
        stacks = self.arrays.copy()
        stacks["stack"] = [deepcopy(self.stack) for _ in range(len(self.arrays))]
        for n in range(len(self.arrays)):
            # stack=deepcopy(self.stack)
            for col in range(len(self.arrays.columns)):
                stacks.loc[n, "stack"].single_param(
                    self.ind_vars.loc[col, "layer"],
                    self.ind_vars.loc[col, "param"],
                    [self.arrays.iloc[n, col], self.ind_vars.loc[col, "unit"],],
                )
            # matrix = [
            #     pd.concat(
            #         [matrix[lay], stacks.loc[n,'stack'].layers[lay].layer_attr],
            #         axis=0,
            #         ignore_index=True,
            #     )
            #     for lay in range(len(matrix))
            # ]
        # [[stack.layers[lay]['btt']for stack in stacks['stack']] for lay in range(len(self.stack.layers))]
        self._stacks = stacks
        # return matrix

    def conv_to_num(self, var_df, res_df):
        unitless = ["resistivity", "dif", "er", "cap", "charge"]
        for par in var_df["param"]:
            if par not in unitless:
                res_df[par] = [
                    res_df.loc[n, par][
                        var_df["unit"][var_df["param"] == par].to_list()[0]
                    ]
                    for n in res_df.index
                ]
        return res_df

    def result(self, layer=None, param=None, unit=None, as_df=False):
        """ log scales: efield """
        # name = self.pickle_check()
        # if name != 'error':
        #     self.sim_name = name
        if layer is not None:
            self.dep_vars["layer"] = layer
        if param is not None:
            self.dep_vars["param"] = param
        if unit is not None:
            self.dep_vars["unit"] = unit
        attrs = self.stack.layers[
            self.dep_vars.loc[0, "layer"]
        ].layer_attr.columns.to_list()
        df_dict = {
            att: [
                stack.layers[self.dep_vars.loc[0, "layer"]][att]
                for stack in self.stacks["stack"]
            ]
            for att in attrs
        }
        df = pd.DataFrame(df_dict)

        attrs_sys = self.stack.sys_info.columns.to_list()
        df_sys_dict = {
            att: [stack[att] for stack in self.stacks["stack"]] for att in attrs_sys
        }
        df_sys = pd.DataFrame(df_sys_dict)

        df = pd.concat([df, df_sys], axis=1)

        df = self.conv_to_num(self.dep_vars, df)

        lay_names = [self.stack.layers[lay].material for lay in self.ind_vars["layer"]]
        ind_var_names = [
            lay_names[n] + "_" + self.ind_vars.loc[n, "param"]
            for n in range(len(self.ind_vars))
        ]

        df[ind_var_names] = self.arrays.to_numpy()
        # self.pickle_save(False)

        if as_df:
            return df.set_index(ind_var_names)
        else:
            return df.set_index(ind_var_names).to_xarray()[
                self.dep_vars.loc[0, "param"]
            ]


#%% Init Prams

# slg_params = [
#     ["soda", {"resis": 2, "thick": "metric"}],
#     ["eva", {"resis": 1, "dif": "max", "thick": "common"}],
#     ["sinx", {"resis": 4, "dif": 3}],
# ]

# slg_wilson_params = [
#     ["soda", {"resis": 2, "thick": "metric"}],
#     ["eva", {"resis": 1, "dif": "max", "thick": "common"}],
#     ["sinx", {"resis": 4, "dif": 2}],
# ]

# slg_vidr_params = [
#     ["soda", {"resis": 1, "thick": "metric"}],
#     ["eva", {"resis": 1, "dif": "max", "thick": "common"}],
#     ["sinx", {"resis": 4, "dif": 3}],
# ]

# slg_vidr_wilson_params = [
#     ["soda", {"resis": 1, "thick": "metric"}],
#     ["eva", {"resis": 1, "dif": "max", "thick": "common"}],
#     ["sinx", {"resis": 4, "dif": 2}],
# ]

# slg_ti_bar_params = [
#     ["soda", {"resis": 1, "thick": "metric"}],
#     ["tio2", {"resis": 1}],
#     ["eva", {"resis": 1, "dif": "max", "thick": "common"}],
#     ["sinx", {"resis": 1, "dif": 2}],
# ]

test_mod = [
    ["soda", {"resis": 2, "thick": "metric"}],
    ["eva", {"resis": 1, "dif": "max", "thick": "common"}],
    ["sinx", {"resis": 4, "dif": 3}],
    ["si", {}],
]

al_eva_params = [
    ["al", {"thick": "metric"}],
    ["eva", {"resis": 1, "dif": "max", "thick": "common"}],
    ["al", {"thick": "metric"}],
]


#%% Init Vars

# temp_var = {"param": "temp", "layer": 1, "unit": "C", "array": [25, 100]}
# efield_var = {"param": "efield", "layer": 1, "unit": "V", "array": [1e3, 1e6]}
# res_e_var = {"param": "resistance", "layer": 1, "unit": "Ohm", "array": [1e8, 1e14]}
# res_g_var = {"param": "resistance", "layer": 0, "unit": "Ohm", "array": [1e8, 1e14]}
# res_n_var = {"param": "resistance", "layer": 0, "unit": "Ohm", "array": [1e8, 1e14]}
# resist_g_var = {"param": "resistivity", "layer": 0, "unit": "Ohm", "array": [1e10, 1e16]}
# resist_e_var = {"param": "resistivity", "layer": 1, "unit": "Ohm", "array": [1e10, 1e16]}
# resist_n_var = {"param": "resistivity", "layer": 2, "unit": "Ohm", "array": [1e10, 1e16]}
# sys_volt_var = {"param": "sys_volt", "layer": 1, "unit": "V", "array": [10, 2000]}
# thick_var = {"param": "thick", "layer": 1, "unit": "um", "array": [100, 600]}
# dif_e_var = {"param": "dif", "layer": 1, "unit": "", "array": [1e-18, 1e-12]}
# dif_n_var = {"param": "dif", "layer": 2, "unit": "", "array": [1e-18, 1e-12]}


# res_var = [res_e_var, res_g_var]
# resist_var = [resist_e_var, resist_g_var]
# resist_varn = [resist_e_var, resist_n_var]

#%% Simulations
single_stack = VoltDev(
    test_mod, 1000, temp=25, area=4 ** 2, sourceC=1e19, conc=1e16, btt_solver="np"
)
# single_stack = VoltDev(al_eva_params, 1000, temp=25, area=4**2, sourceC=1e19, conc=1e16, btt_solver='np')
# triple_stack = VoltDev(al_eva_params, 1000, temp=25, area=4**2, sourceC=1e19, conc=1e16, btt_solver='np')
# thick_stack = VoltDev(al_eva_params, 1000, temp=25, area=4**2, sourceC=1e19, conc=1e16, btt_solver='np')
# thin_stack = VoltDev(al_eva_params, 1000, temp=25, area=4**2, sourceC=1e19, conc=1e16, btt_solver='np')

# triple_stack[1]["thick"] = triple_stack[1]["thick"]*3
# thick_stack[1]["thick"] = up.Length(.58,"mm") -  thick_stack[0]["thick"] * 2
# thick_stack[1]["thick"] = up.Length(1.58,"mm") -  thick_stack[0]["thick"] * 2

# rhoe_rhog = Module(slg_vidr_params, 1000, 10, [resist_e_var, resist_g_var], temp=60, sourceC=1e19, conc=1e16, btt_solver='np')
# btt1 = rhoe_rhog.result(layer=1, as_df=True)
# x1 = btt1['btt']
# btt2 = rhoe_rhog.result(layer=2, as_df=True)
# x2 = btt2['btt']

# rhoe_rhon = Module(slg_vidr_params, 1000, 50, [resist_e_var, resist_n_var], temp=60, sourceC=1e19, conc=1e16, btt_solver='np')
# btt3 = rhoe_rhon.result(layer=1, as_df=True)
# x3 = btt3['btt']
# btt4 = rhoe_rhon.result(layer=2, as_df=True)
# x4 = btt4['btt']

# rhon_rhog = Module(slg_vidr_params, 1000, 50, [resist_n_var, resist_g_var], temp=60, sourceC=1e19, conc=1e16, btt_solver='np')
# btt5 = rhon_rhog.result(layer=1, as_df=True)
# x5 = btt5['btt']
# btt6 = rhon_rhog.result(layer=2, as_df=True)
# x6 = btt6['btt']


# rhoe_dife_s = Module(slg_params, 1000, 25, [resist_e_var, resist_g_var, dif_e_var], temp=60, sourceC=1e19, conc=1e16, btt_solver='np')
# btt7 = rhoe_dife_s.result(layer=1, as_df=True)
# x7 = btt7['btt']
# y1 = x7[x7.index.get_level_values(0) == x7.index.get_level_values(1)]

# rhoe_difn_s = Module(slg_params, 1000, 25, [resist_e_var, resist_g_var, dif_n_var], temp=60, sourceC=1e19, conc=1e16, btt_solver='np')
# btt8 = rhoe_difn_s.result(layer=2, as_df=True)
# x8 = btt8['btt'][x7.index.get_level_values(0) == x7.index.get_level_values(1)]
# y2 = x8[x8.index.get_level_values(0) == x8.index.get_level_values(1)]

# rhoe_dife_v = Module(slg_vidr_params, 1000, 50, [resist_e_var, dif_e_var], temp=60, sourceC=1e19, conc=1e16, btt_solver='np')
# btt9 = rhoe_dife_v.result(layer=1, as_df=True)
# x9 = btt9['btt']

# rhoe_difn_v = Module(slg_vidr_params, 1000, 50, [resist_e_var, dif_n_var], temp=60, sourceC=1e19, conc=1e16, btt_solver='np')
# btt10 = rhoe_difn_v.result(layer=2, as_df=True)
# x10 = btt10['btt']


# # slg_dif_res2 = Module(slg_vidr_params, 1000, 50, [resist_e_var, dif_var], temp=60, sourceC=1e19, conc=1e16, btt_solver='char')
# # btt_dif_res2 = slg_dif_res2.result(layer=2, as_df=True)


# # efield_vidr_slg_resis = vidr_slg_resis.result(2, 'efield','V').to_dataframe('value')
# # btt_vidr_slg_resis = vidr_slg_resis.result(2, 'btt', 's').to_dataframe('value')

# # print_1d = btt_vidr_slg_resis.to_dataframe('value')
# # print_2d = btt_init_slg.to_dataframe('value').unstack("dim_0")
