# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:42:38 2021

@author: j2cle
"""

# %% Imports
import os
import numpy as np
import pandas as pd

import utilities as ut

# import sympy as sym
from itertools import product

from scipy import optimize

from dataclasses import astuple
from copy import deepcopy


# def xarr_to_dict
KB_EV = 8.617333262145e-5


# %% Functions
def fit_module(
    params, ind_vars, target="time", size=20, cols=None, name="", layers=None, **mod_kwargs
):
    """Calculate. generic discription."""
    std_dict = dict(
        sys_volt=1500,
        temp=85,
        source=1e19,
        conc=1e16,
        np_type="np",
    )

    mod_kwargs = {**std_dict, **mod_kwargs}

    res = {}
    # final = {}
    # module1 = VoltDev(params, 1500, temp=25)
    mod1 = Module(
        params,
        size,
        ind_vars,
        np_target=target,
        **mod_kwargs,
    )

    if layers is None or "gls" in layers:
        res[f"gls{name}"] = mod1.result(layer=0, as_df=True)
    if layers is None or "enc" in layers:
        res[f"enc{name}"] = mod1.result(layer=1, as_df=True)
    if layers is None or "arc" in layers:
        res[f"arc{name}"] = mod1.result(layer=2, as_df=True)

    if cols is None:
        return res

    final = res["enc"][["sys_time", "sys_res", "sys_volt"]]
    for key in res.keys():
        for col in cols:  # ["dif", "time", "efield", "volt", "resistance"]:
            final[f"{key}_{col}"] = res[key][col]

    return final


def vary_cond_fits(
    params,
    ind_vars,
    mats=["eva", "poe"],
    mat_lays=1,
    mat_var="resis",
    target="time",
    size=50,
    layers=None,
    cols=None,
    verbose=False,
    **mod_kwargs,
):
    """Calculate. generic discription."""
    res = {}
    for mat in mats:
        for label in materials.database.loc[
            (mat, mat_var), "note"
        ]:  # ["Kapur2015", "Kapur2015, PID-res"]:
            params[mat_lays][0] = mat
            params[mat_lays][1][mat_var] = label
            if verbose:
                print(f"{mat}_{label}")

            if cols is None:
                res_temp = fit_module(
                    params,
                    ind_vars,
                    target,
                    size,
                    name=f"_{mat}_{label}",
                    layers=layers,
                    **mod_kwargs,
                )
                res = {**res, **res_temp}
            else:
                res[f"{mat}_{label}"] = fit_module(
                    params, ind_vars, target, size, cols=cols, **mod_kwargs
                )
    return res


def single_var(df_dict, col, reject=[]):
    """Calculate. generic discription."""
    df = None
    for key in df_dict.keys():
        store = 1
        for rej in reject:
            if rej in key:
                store = 0
        if store:
            if df is None:
                df = pd.DataFrame(index=df_dict[key].index)

            df[key] = df_dict[key][col]
    return df


# %% Classes
class NernstPlanck(object):
    """Calculate. generic discription."""

    def __init__(self, df, data=None):
        self.df = df
        self.data = data
        self.status = "None"

    def char_eq(self, target="time", as_set=False):
        """Return sum of squared errors (pred vs actual)."""
        param_values = self.df.loc[0, :].to_dict()
        param_values["boltz"] = KB_EV

        char = "2*(dif*time)**(1/2)+dif/(boltz*temp)*efield*time-thick"
        if target == "depth":
            char = "2*(dif*time)**(1/2)+dif/(boltz*temp)*efield*time-depth"

        # currently non functional and bypassed in layer, issue with sympy
        # if target == "temp" and self.data is not None:
        # param_values.pop("dif")
        # param_values["difnaugt"] = self.data.var1["dif"]
        # param_values["acten"] = self.data.var2["dif"]
        # char = "2*(difnaugt*exp(-acten/(boltz*temp))*time)**(1/2)
        # +difnaugt*exp(-acten/(boltz*temp))/(boltz*temp)*efield*time-thick"

        x0 = ut.eqn_sets(param_values, target=target, eqns=char, as_set=False)
        self.status = "sympy char"
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
        df = self.df.copy()
        if target == "temp":
            x0 = optimize.least_squares(
                ut.char_time,
                args=(
                    self.df.copy(),
                    self.data.copy(),
                    target,
                    scale,
                ),
                x0=300,
                bounds=(0, 10000),
                **ls_params,
            ).x[0]
            x0_status = "ls char"
        else:
            x0 = self.char_eq(target, **pre_kwargs)
            x0_status = "sympy char"

        if "log" in scale.lower():
            df[target] = np.log10(df[target])
            x0 = np.log10(x0)

        bounds = (x0 * (1 - bound), x0 * (1 + bound))
        val = {**{"x0": x0, "bounds": (min(bounds), max(bounds))}, **ls_params}

        if ratio is None and "source" in self.df.columns:
            ratio = self.df.conc / self.df.source
        else:
            ratio = 0.08
        try:
            results = optimize.least_squares(
                ut.np_cost,
                args=(
                    df,
                    self.data.copy(),
                    ratio,
                    target,
                    scale,
                ),
                **val,
            ).x[0]
            self.status = "ls np"
        except ValueError:
            results = 0
            self.status = "ls ValueError"

        if "log" in scale.lower():
            results = 10**results

        if results == x0:
            self.status = f"ls fail; {x0_status}"

        return results


class MatDatabase(object):
    """Calculate. generic discription."""

    # def __init__(self):
    path = ut.pathify("work", "Data", "Databases")
    file = "material_data.xlsx"
    database = pd.read_excel(os.sep.join((path, file)), index_col=[0, 1, 2])

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
        """Calculate. generic discription."""
        self.material = material.lower()
        self.data = self.mat_data.xs(1, level="label").copy()
        labels = {"resis": resis, "perm": perm, "dif": dif, "thick": thick}

        if not all(val == 1 for val in labels.values()):
            self.change_data(labels)
        return self.data

    def change_data(self, labels):
        """Calculate. generic discription."""
        for key, val in labels.items():
            if isinstance(val, list):
                self.data.loc[key, "var1":] = val
            else:
                if not isinstance(val, int):
                    val = self.note_to_label(key, val)
                self.data.loc[key, :] = self.mat_data.loc[(key, val), :]

    def note_to_label(self, info, label):
        """Calculate. generic discription."""
        data = self.mat_data.loc[info, :]
        try:
            if isinstance(label, (list, tuple, np.ndarray)):
                for lab in label:
                    data = data[data.note.str.contains(lab)]
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
        """Calculate. generic discription."""
        label = self.database.loc[(material, info), :].index.max() + 1
        self.database.loc[(material, info, label), :] = [note, var1, var2]

    def save(self):
        """Calculate. generic discription."""
        self.database.to_excel(os.sep.join((self.path, self.file)))


class Layer(object):
    """Holds layer material information.  Calculates the related datapoints."""

    def __init__(
        self,
        material="mat",
        data_guide={"resis": 1, "perm": 1, "dif": 1, "thick": 1},
        temp=25,
        efield=0,
        volt=0,
        area=1,
        np_type="char_time",
        np_target="btt",
        rt_link=False,
        dt_link=False,
        cp_link=False,
        temp_threshold=150,
        **kwargs,
    ):
        """Return layer object."""
        # mat_test1=MatDatabase()
        # Initialize logic related values
        self.np_type = np_type
        self.np_target = np_target
        self.btt = True
        self.rt_link = rt_link
        self.dt_link = dt_link
        self.cp_link = cp_link
        self.temp_threshold = temp_threshold
        self.status = "None"

        # Initialize material related values
        self.material = material

        # Import data
        self.data = MatDatabase().get_df(self.material, **data_guide)

        # Initialize np related values
        self.temp = temp
        self.depth = self.thick
        self.time = ut.Time(4, "d", 1, "s")
        if efield == 0 and volt != 0:
            self.volt = volt
        else:
            self.efield = efield

        # Initialize other values
        self.area = area

        [setattr(self, key, val) for key, val in kwargs.items()]

    def __getitem__(self, item):
        """Return sum of squared errors (pred vs actual)."""
        return getattr(self, item)

    def __setitem__(self, item, val):
        """Return sum of squared errors (pred vs actual)."""
        setattr(self, item, val)

    # data (arrh) related properties
    @property
    def thick(self):
        """Return thickness of layer."""
        if self.np_target == "thick":
            self.np_solver("thick")
        return self._thick

    @thick.setter
    def thick(self, val):
        if val is not None:
            self.parse_arrh("thick", val, "cm")

    @property
    def _thick(self):
        """Return thickness of layer."""
        if self.data.var2["thick"] != 0:
            var = ut.arrh(temp=self._temp.K, *self.data.loc["thick", ["var1", "var2"]])
        else:
            var = self.data.var1["thick"]
        return ut.Length(var, self.data.unit["thick"], 1, "cm")

    @property
    def resistivity(self):
        """rho = rho0 * exp(Ea / k_B T)."""
        return self._resistivity

    @resistivity.setter
    def resistivity(self, val):
        """Set rho_0 to val."""
        if val is not None:
            self.parse_arrh("resis", val, "Ohm.cm")
            if self.rt_link and self.data.var2["resis"] != 0:
                self.temp = ut.Temp(
                    ut.arrh(result=self._resistivity, *self.data.loc["resis", ["var1", "var2"]]),
                    "K",
                    1,
                    "C",
                )

    @property
    def _resistivity(self):
        """rho = rho0 * exp(Ea / k_B T)."""
        return ut.arrh(temp=self._temp.K, *self.data.loc["resis", ["var1", "var2"]])

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
            self.parse_arrh("perm", val, "F/cm")
            if self.data.var1["perm"] <= 1:
                self.data.var1["perm"] = 1

    @property
    def _er(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.data.var2["perm"] != 0:
            return ut.arrh(temp=self._temp.K, *self.data.loc["perm", ["var1", "var2"]])
        else:
            return self.data.var1["perm"]

    @property
    def dif(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.np_target == "dif":
            self.np_solver("dif", scale="log")
        return self._dif

    @dif.setter
    def dif(self, val):
        if val is not None:
            self.parse_arrh("dif", val, "cm2/s")
            if self.dt_link and self.data.var2["dif"] != 0:
                self.temp = ut.Temp(
                    ut.arrh(result=self._dif, *self.data.loc["resis", ["var1", "var2"]]),
                    "K",
                    1,
                    "C",
                )

    @property
    def _dif(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.data.var2["dif"] != 0:
            return ut.arrh(temp=self._temp.K, *self.data.loc["dif", ["var1", "var2"]])
        else:
            return self.data.var1["dif"]

    # singular value properties
    @property
    def depth(self):
        """Return thickness of layer."""
        if self.np_target == "depth":
            self.np_solver("depth")
        elif self.btt:
            return self._thick
        return self._depth

    @depth.setter
    def depth(self, val):
        if val is not None:
            self._depth = self.parse_unit(ut.Length, val, "cm")

    @property
    def time(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.np_target == "time" or self.np_target == "btt":
            self.np_solver("time", scale="log")
        return self._time

    @time.setter
    def time(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            self._time = self.parse_unit(ut.Time, val, "s")

    @property
    def temp(self):
        """Return thickness of layer."""
        if self.np_target == "temp":
            self.np_solver("temp")
        return self._temp

    @temp.setter
    def temp(self, val):
        if val is not None:
            self._temp = self.parse_unit(ut.Temp, val, "C")
            if float(str(self._temp)) >= self.temp_threshold:
                self._temp = self.parse_unit(ut.Temp, val, "K", 1, "C")

    @property
    def efield(self):
        """Return V/cm as volt type asuming cm's."""
        if self.np_target == "efield":
            self.np_solver("efield", scale="log")
        return self._efield

    @efield.setter
    def efield(self, val):
        """Ensure input is a Volt class."""
        if val is not None:
            self._efield = self.parse_unit(ut.Volt, val, "V")

    @property
    def area(self):
        """Return Area."""
        return self._area

    @area.setter
    def area(self, val):
        if val is not None:
            self._area = self.parse_unit(ut.Length, val, "cm", 2)
            if self._area.exp != 2:
                self._area = ut.Length(self._area.value**2, self._area.unit, 2, "cm")

    # calculated properties
    @property
    def volt(self):
        """V = E * t."""
        return ut.Volt(self.efield.V * self.thick.cm)

    @volt.setter
    def volt(self, val):
        """E = V / t."""
        if val is not None:
            self.efield = val / self.thick.cm

    @property
    def resistance(self):
        """R = rho * t / A."""
        return ut.Res(self.resistivity * self.thick.cm / self.area.cm)

    @resistance.setter
    def resistance(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            resis = self.parse_unit(ut.Res, val, "Ohm")
            self.resistivity = resis.Ohm * self.area.cm / self.thick.cm

    @property
    def curr(self):
        """Return sum of squared errors (pred vs actual)."""
        return ut.Curr(self.volt.V / self.resistance.Ohm)

    @curr.setter
    def curr(self, val):
        """E = I * R / t."""
        if val is not None:
            self.efield = val * self.resistance.Ohm / self.thick.cm

    @property
    def cap(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.er * ut.Length(ut.PERM, "m", -1).cm * self.area.cm / self.thick.cm

    @cap.setter
    def cap(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            self.er = val * self.thick.cm / (self.area.cm * ut.Length(ut.PERM, "m", -1).cm)

    @property
    def charge(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.er * ut.Length(ut.PERM, "m", -1).cm * self.resistivity * self.curr.A

    @charge.setter
    def charge(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            self.er = val / (ut.Length(ut.PERM, "m", -1).cm * self.resistivity * self.curr.A)

    @property
    def source(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_source"):
            self._source = 1e21
        if self.cp_link:
            return ut.poisson(self.er, volt=self.volt.V, thick=self.thick.cm) / 10
        else:
            return self._source

    @source.setter
    def source(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            self._source = val

    @property
    def conc(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_conc"):
            self._conc = 1e10
        if self._conc > self.source:
            return self.source / 100
        else:
            return self._conc

    @conc.setter
    def conc(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            self._conc = val

    # dataframe properties
    @property
    def info_electrical(self):
        """Return the necessary information."""
        df = {
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
        return pd.DataFrame([df])

    @property
    def info_np(self):
        """Return the necessary information."""
        try:
            df = {
                "time": float(self._time.s),
                "depth": float(self._depth.cm),
                "thick": float(self._thick.cm),
                "dif": float(self._dif),
                "temp": float(self._temp.K),
                "efield": float(self._efield.V),
                "source": float(self.source),
                "conc": float(self.conc),
            }
        except AttributeError:
            df = {}
        return pd.DataFrame([df])

    @property
    def info_attrs(self):
        """Return the necessary information."""
        df = {
            "material": str(self.material),
            "dif": (self.dif),
            "thick": (self.thick),
            "depth": (self.depth),
            "time": (self.time),
            "temp": (self.temp),
            "efield": (self.efield),
            "resistivity": (self.resistivity),
            "volt": (self.volt),
            "resistance": (self.resistance),
            "curr": (self.curr),
            "er": (self.er),
            "cap": (self.cap),
            "charge": (self.charge),
            "source": (self.source),
            "conc": (self.conc),
            "np_type": (self.np_type),
        }
        return pd.DataFrame([df])

    @property
    def base_units(self):
        """Return the necessary information."""
        df = {
            "thick": "cm",
            "depth": "cm",
            "time": "s",
            "temp": "C",
            "efield": "V",
            "volt": "V",
            "resistance": "Ohm",
            "curr": "A",
        }
        return df

    def np_solver(self, target, source=None, conc=None, scale="lin", ls_params={}, **pre_kwargs):
        """Return sum of squared errors (pred vs actual)."""
        if target in self.info_np.columns and self._dif != 0:
            if source is not None:
                self.source = source
            if conc is not None:
                self.conc = conc
            ratio = self.conc / self.source
            nernst = NernstPlanck(self.info_np, self.data)
            if "np" in self.np_type or target == "temp":
                # res = nernst.np_sim(target, ratio, scale, ls_params, **pre_kwargs)
                self[target] = nernst.np_sim(target, ratio, scale, ls_params, **pre_kwargs)
            else:
                self[target] = nernst.char_eq(target, **pre_kwargs)
            self.status = nernst.status
        else:
            return

    def parse_unit(self, unit_class, val, *args):
        """Calculate. generic discription."""
        if isinstance(val, unit_class):
            return val
        elif isinstance(val, (tuple, list)):
            return unit_class(*val)
        elif isinstance(val, (dict)):
            return unit_class(**val)
        else:
            return unit_class(val, *args)

    def parse_arrh(self, param, *args):
        """Calculate. generic discription."""
        args = list(args)
        note = "manual"
        unit = ""
        if isinstance(args[0], str):
            note = args.pop(0)
        if isinstance(args[-1], str):
            unit = args.pop(-1)

        if isinstance(args[0], ut.BaseUnits):
            self.data.loc[param, :] = [note, float(str(args[0])), 0, args[0].print_unit]
        elif isinstance(args[0], (tuple, list)):
            if isinstance(args[0], tuple):
                args[0] = list(args[0])
            if isinstance(args[0][-1], str):
                unit = args[0].pop(-1)
            if len(args[0]) > 1:
                self.data.loc[param, :] = [note, args[0][0], args[0][1], unit]
            else:
                self.data.loc[param, :] = [note, args[0][0], 0, unit]
        elif isinstance(args[0], (dict)):
            self.data.loc["dif", args[0].keys()] = args[0].values()
        else:
            self.data.loc[param, :] = [note, args[0], 0, unit]


class Stack:
    """Calculate. generic discription."""

    def __init__(
        self,
        layers=[["boro", {}], ["eva", {}], ["sinx", {}]],
        **layer_kwargs,
    ):

        if isinstance(layers, (int, np.integer)):
            self.layers = [Layer() for _ in range(layers)]
        elif all(isinstance(lay, Layer) for lay in layers):
            self.layers = layers
        else:
            self.layers = [
                Layer(material=mat, data_guide=guide, **layer_kwargs) for mat, guide in layers
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
        if isinstance(items[0], ut.BaseUnits):
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
            if isinstance(val, (int, float, np.integer, np.float)) or len(val) != len(self.layers):
                [setattr(lay, item, val) for lay in self.layers]
            else:
                [setattr(self.layers[n], item, val[n]) for n in range(len(self.layers))]
        elif hasattr(self, item):
            setattr(self, item, val)

    @property
    def uniform_vars(self):
        """Calculate. generic discription."""
        return ["temp", "curr", "area"]

    @property
    def stack_info(self):
        """Return sum of squared errors (pred vs actual)."""
        df = pd.concat([lay.info_electrical for lay in self.layers], axis=0, ignore_index=True)
        return df

    @property
    def stack_np(self):
        """Return sum of squared errors (pred vs actual)."""
        df = pd.concat([lay.info_np for lay in self.layers], axis=0, ignore_index=True)
        return df

    @property
    def stack_data(self):
        """Return sum of squared errors (pred vs actual)."""
        df = pd.concat(
            [lay.data.loc[:, "note"] for lay in self.layers], axis=1, ignore_index=True
        ).T

        return df


class VoltDev(Stack):
    """Calculate. generic discription."""

    def __init__(self, layers, sys_volt=0, **layer_kwargs):

        if isinstance(layers, Stack):
            self.layers = layers.layers
        elif isinstance(layers, (int, np.integer)):
            self.layers = [Layer() for _ in range(layers)]
        elif all(isinstance(lay, Layer) for lay in layers):
            self.layers = layers
        else:
            self.layers = [
                Layer(material=mat, data_guide=guide, **layer_kwargs) for mat, guide in layers
            ]
        self.sys_volt = sys_volt

    @property
    def sys_time(self):
        """Calculate. generic discription."""
        # return ut.Res(self.stack.resistance.Ohm.sum())
        return ut.Time(self["time"].s.sum())

    @property
    def sys_res(self):
        """Calculate. generic discription."""
        # return ut.Res(self.stack.resistance.Ohm.sum())
        return ut.Res(self["resistance"].Ohm.sum())

    @property
    def sys_volt(self):
        """Calculate. generic discription."""
        if not hasattr(self, "_sys_volt"):
            self._sys_volt = ut.Volt(1)
        return self._sys_volt

    @sys_volt.setter
    def sys_volt(self, val):
        if isinstance(val, ut.Volt):
            self._sys_volt = val
        elif isinstance(val, (tuple, list)):
            self._sys_volt = ut.Volt(*val)
        elif isinstance(val, (dict)):
            self._sys_volt = ut.Volt(**val)
        else:
            self._sys_volt = ut.Volt(val, "V")
        self["volt"] = self["resistance"].Ohm / self.sys_res.Ohm * self._sys_volt.V

    @property
    def sys_cap(self):
        """Return sum of squared errors (pred vs actual)."""
        return ut.inv_sum_invs(self["cap"])

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
            "sys_time": self.sys_time,
            "sys_res": self.sys_res,
            "sys_volt": self.sys_volt,
            "sys_cap": self.sys_cap,
            "sys_charge": self.sys_charge,
            "ave_charge": self.ave_charge,
        }
        return pd.DataFrame([df])

    @property
    def sys_units(self):
        """Return the necessary information."""
        df = {
            "sys_time": "s",
            "sys_res": "Ohm",
            "sys_volt": "V",
        }
        return {**self.layers[0].base_units, **df}

    def single_param(self, layer=0, param="volt", val=0, unit=None, lock_ext=False):
        """Update single layer information  and find needed system volt to accomodate this."""
        ext = self.sys_volt.V
        r_tot = self.sys_res.Ohm
        if isinstance(unit, str):
            val = [val, unit]
        if hasattr(self, param) or param in self.uniform_vars:
            self[param] = val
        else:
            self[layer][param] = val
            if r_tot != self.sys_res.Ohm or lock_ext:
                self.sys_volt = ut.Volt(ext, "V")
            else:
                self.sys_volt = (
                    self[layer]["volt"].V * self.sys_res.Ohm / self[layer]["resistance"].Ohm
                )
        return

    def curr_eval(self, current=0, lay_params={}, alter_volt=False):
        """Calculate needed R to explain current and related eff_volt."""
        if isinstance(current, ut.Curr):
            tar_curr = current
        else:
            tar_curr = ut.Curr(current)
        if lay_params == {}:
            lay_params = {"material": "air", "temp": self["temp"], "area": self["area"]}
        added_layer = Layer(**lay_params)
        added_layer.volt = tar_curr.A * added_layer.resistance.Ohm

        eff_volt = ut.Volt(self.sys_volt.V - added_layer.volt.V)

        if alter_volt:
            self.sys_volt = eff_volt
        return {"eff_volt": eff_volt, "layer": added_layer}

    def cap_eval(
        self,
        dampener=0.05,
        iters=100,
        qtol=0.05,
    ):
        """Calculate. generic discription."""
        n = 0
        dev = 1
        while dev > qtol and n < iters:
            eps0 = np.array(self["er"])
            eps1 = self.sys_charge / (
                ut.Length(ut.PERM, "m", -1).cm * self["efield"].V * self["area"].cm
            )
            eps_dev = eps0 - eps1
            epsf = eps0 - eps_dev * dampener

            self[abs(eps_dev / eps0).argmax()]["er"] = epsf[abs(eps_dev / eps0).argmax()]
            dev = max(abs(self.ave_charge - np.array(self["charge"])) / self.ave_charge)
            n += 1
        return


class Module(object):
    """Calculate. generic discription."""

    def __init__(
        self,
        stack,
        size=100,
        ind_vars=[],
        dep_vars=[],
        sys_volt=0,
        **kwargs,
    ):

        if not isinstance(stack, VoltDev):
            stack = VoltDev(stack, sys_volt, **kwargs)
        self.stack = stack
        self.sys_volt = sys_volt
        self.size = size

        self.path = ut.pathify("work", "Python Scripts", "Active", "Database")
        self.folder = "MigrationModules"

        if dep_vars == []:
            dep_vars = [
                {"param": "time", "layer": 1, "unit": "s"},
            ]
        self.dep_vars = dep_vars

        if ind_vars == []:
            ind_vars = [
                {"param": "temp", "layer": 1, "unit": "C", "array": [25, 100]},
                {"param": "efield", "layer": 1, "unit": "V", "array": [1e3, 1e6]},
            ]
        self.ind_vars = ind_vars

        self.stacks
        # self.solver(**self.ind_vars.loc[0, :].to_dict())
        # if "y" in self.ind_vars.keys():
        #     self.solver(**self.ind_vars["y"])

    def __getitem__(self, item):
        """Return sum of squared errors (pred vs actual)."""
        return getattr(self, item)

    @property
    def ind_vars(self):
        """Store vars. log scales: efield."""
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
        """Store vars. log scales: efield."""
        return self._dep_vars

    @dep_vars.setter
    def dep_vars(self, val):
        if not isinstance(val, list):
            self._dep_vars = pd.DataFrame([val])
        else:
            self._dep_vars = pd.DataFrame(val)

    @property
    def arrays(self):
        """Store vars. log scales: efield."""
        return self._arrays

    @property
    def sim_name(self):
        """Store vars. log scales: efield."""
        if not hasattr(self, "_sim_name"):
            x = "_".join(mat for mat in self.stack.stack_info["mat"])
            y = "_".join(par for par in self.ind_vars["param"])
            self._sim_name = "_".join((x, y))
        return self._sim_name

    @sim_name.setter
    def sim_name(self, name):
        self._sim_name = name

    @property
    def stacks(self):
        """Store vars. log scales: efield."""
        if not hasattr(self, "_stacks"):
            self.solver()
            # self.pickle_save()
        return self._stacks

    def range_gen(self, min_max, scale=None):
        """Calculate. generic discription."""
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
        """Calculate. generic discription."""
        stacks = self.arrays.copy()
        stacks["stack"] = [deepcopy(self.stack) for _ in range(len(self.arrays))]
        for n in range(len(self.arrays)):
            # stack=deepcopy(self.stack)
            for col in range(len(self.arrays.columns)):
                stacks.loc[n, "stack"].single_param(
                    self.ind_vars.loc[col, "layer"],
                    self.ind_vars.loc[col, "param"],
                    [
                        self.arrays.iloc[n, col],
                        self.ind_vars.loc[col, "unit"],
                    ],
                )

        self._stacks = stacks

    def conv_to_num(self, var_df, res_df):
        """Calculate. generic discription."""
        unitless = ["resistivity", "dif", "er", "cap", "charge", "source", "conc"]
        for par in var_df["param"]:
            if par not in unitless:
                try:
                    res_df[par] = [
                        res_df.loc[n, par][var_df["unit"][var_df["param"] == par].to_list()[0]]
                        for n in res_df.index
                    ]
                except IndexError:
                    pass
        return res_df

    def result(self, layer=None, param=None, unit=None, as_df=False):
        """Calculate. generic discription."""
        # name = self.pickle_check()
        # if name != 'error':
        #     self.sim_name = name
        if layer is not None:
            self.dep_vars["layer"] = layer
        if param is not None:
            self.dep_vars["param"] = param
        if unit is not None:
            self.dep_vars["unit"] = unit
        attrs = self.stack.layers[self.dep_vars.loc[0, "layer"]].info_attrs.columns.to_list()
        df_dict = {
            att: [
                stack.layers[self.dep_vars.loc[0, "layer"]][att] for stack in self.stacks["stack"]
            ]
            for att in attrs
        }
        df = pd.DataFrame(df_dict)

        attrs_sys = self.stack.sys_info.columns.to_list()
        df_sys_dict = {att: [stack[att] for stack in self.stacks["stack"]] for att in attrs_sys}
        df_sys = pd.DataFrame(df_sys_dict)

        df = pd.concat([df, df_sys], axis=1)

        df = self.conv_to_num(self.dep_vars, df)

        lay_names = [self.stack.layers[lay].material for lay in self.ind_vars["layer"]]
        ind_var_names = [
            lay_names[n] + "_" + self.ind_vars.loc[n, "param"] for n in range(len(self.ind_vars))
        ]

        df[ind_var_names] = self.arrays.to_numpy()
        # self.pickle_save(False)

        if as_df:
            df = self.conv_to_num(
                pd.DataFrame(self.stack.sys_units.items(), columns=["param", "unit"]), df
            )
            return df.set_index(ind_var_names)
        else:
            return df.set_index(ind_var_names).to_xarray()[self.dep_vars.loc[0, "param"]]


# %% Init Prams
var_dict = dict(
    temp_var={"param": "temp", "layer": 1, "unit": "C", "array": [20, 100]},
    efield_var={"param": "efield", "layer": 1, "unit": "V", "array": [1e2, 1e6]},
    efield_n_var={"param": "efield", "layer": 2, "unit": "V", "array": [1e3, 1e6]},
    volt_var={"param": "volt", "layer": 1, "unit": "V", "array": [1, 1e4]},
    sys_volt_var={"param": "sys_volt", "layer": 1, "unit": "V", "array": [1, 1e4]},
    res_e_var={"param": "resistance", "layer": 1, "unit": "Ohm", "array": [1e8, 1e14]},
    res_g_var={"param": "resistance", "layer": 0, "unit": "Ohm", "array": [1e8, 1e14]},
    res_n_var={"param": "resistance", "layer": 2, "unit": "Ohm", "array": [1e8, 1e14]},
    resist_g_var={"param": "resistivity", "layer": 0, "unit": "Ohm.cm", "array": [1e10, 1e16]},
    resist_e_var={"param": "resistivity", "layer": 1, "unit": "Ohm.cm", "array": [1e10, 1e16]},
    resist_n_var={"param": "resistivity", "layer": 2, "unit": "Ohm.cm", "array": [1e10, 1e16]},
    thick_var={"param": "thick", "layer": 1, "unit": "um", "array": [100, 600]},
    dif_e_var={"param": "dif", "layer": 1, "unit": "", "array": [1e-18, 1e-12]},
    dif_n_var={"param": "dif", "layer": 2, "unit": "", "array": [1e-18, 1e-12]},
)

params = [
    ["soda", {"resis": "vidrasa", "thick": "metric"}],
    ["eva", {"resis": "Kapur2015, PID-res", "dif": "max", "thick": "thin"}],
    ["sinx", {"resis": "expected", "dif": "gastrow2020"}],
]

params_boro = [
    ["boro", {"resis": "duran", "thick": "metric"}],
    ["eva", {"resis": "Kapur2015, PID-res", "dif": "max", "thick": "thin"}],
    ["sinx", {"resis": "expected", "dif": "gastrow2020"}],
]

tri_var = [var_dict["resist_g_var"], var_dict["resist_e_var"], var_dict["resist_n_var"]]
btt_var = [var_dict["temp_var"], var_dict["efield_var"]]

path = ut.pathify("work", "Data", "Analysis", "Simulations", "Module")
stacks = ["Glass_Enc_Sinx"]

materials = MatDatabase()

std_kwargs = dict(
    temp=85,
    sys_volt=1500,
    cp_link=True,
    # source=1e19,
    # conc=1e16,
    np_type="np",
)

# # %% single var sims
# enc_temp_dep = vary_cond_fits(
#     params,
#     [var_dict["temp_var"]],
#     mats=["eva", "poe"],
#     mat_lays=1,
#     mat_var="resis",
#     target=None,
#     size=50,
# )
# enc_rho_temp_dep = single_var(enc_temp_dep, "resistivity", ["gls", "arc"])
# gls_temp_dep = vary_cond_fits(
#     params,
#     [var_dict["temp_var"]],
#     mats=["boro", "soda"],
#     mat_lays=0,
#     mat_var="resis",
#     target=None,
#     size=50,
# )
# gls_rho_temp_dep = single_var(gls_temp_dep, "resistivity", ["gls", "arc"])

# %% btt sims
btt = vary_cond_fits(
    params,
    btt_var,
    mats=["eva"],
    mat_lays=1,
    mat_var="resis",
    target="time",
    size=25,
    verbose=True,
    layers=["enc"],
    **{**std_kwargs, **dict(time=0)},
)
ut.save(
    btt,
    os.sep.join((path, stacks[0], "Soda_Varied_low")),
    "temp_e_time4",
)

# btt_boro = vary_cond_fits(
#     params_boro,
#     btt_var,
#     mats=["eva"],
#     mat_lays=1,
#     mat_var="resis",
#     target="time",
#     size=50,
#     verbose=True,
#     layers=["enc"],
#     **{**std_kwargs, **dict(time=0)},
# )
# ut.save(
#     btt_boro,
#     os.sep.join((path, stacks[0], "Boro_Varied_low")),
#     "temp_e_time3",
# )


# %%Tri sims
# btt: set time to 0 via: **{**std_kwargs, **dict(time=0)}, dif: no kwarg update

# params_tri = [
#     ["soda", {"resis": "vidrasa", "thick": "metric"}],
#     ["eva", {"resis": "Kapur2015, PID-res", "dif": "max", "thick": "thin"}],
#     ["sinx", {"resis": "expected", "dif": "lowwest"}],
# ]

# tri_time_v5 = fit_module(
#     params_tri,
#     tri_var,
#     "time",
#     50,
#     cols=["time", "efield"],  #["dif", "time", "efield", "volt", "resistance"],
#     **{**std_kwargs, **dict(time=0)},
# )
# ut.save(
#     tri_time_v5,
#     os.sep.join((path, stacks[0], "Varied_resistivities")),
#     "all_time_v5",
# )

# tri_time_v1 = fit_module(
#     params,
#     tri_var,
#     "time",
#     50,
#     cols=["time", "efield"],  #["dif", "time", "efield", "volt", "resistance"],
#     **{**std_kwargs, **dict(time=0)},
# )
# ut.save(
#     tri_time_v1,
#     os.sep.join((path, stacks[0], "Varied_resistivities")),
#     "all_time_v1",
# )
# tri_times = vary_cond_fits(
#     params_tri,
#     tri_var,
#     mats=["sinx"],
#     mat_lays=2,
#     mat_var="dif",
#     target="time",
#     size=25,
#     cols=["dif", "time", "efield", "volt", "resistance"],
#     **{**std_kwargs, **dict(time=0)},
# )

# ut.save(
#     tri_times,
#     os.sep.join((path, stacks[0], "Varied_resistivities")),
#     "all_times",
# )

# %% diffusion sims
# tri_dif_v1 = fit_module(
#     params,
#     tri_var,
#     "dif",
#     25,
#     cols=None,  #["dif", "time", "efield", "volt", "resistance"],
#     layers=["arc"],
#     **std_kwargs,
# )
# ut.save(
#     tri_dif_v1,
#     os.sep.join((path, stacks[0], "Varied_resistivities")),
#     "all_dif_v1",
# )
