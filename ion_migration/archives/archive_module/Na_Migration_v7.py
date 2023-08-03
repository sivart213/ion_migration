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
from copy import copy, deepcopy

def arrh_wrap(T, pre_fac, E_A):
    T = up.Temp(T, "C").K
    return gf.arrh(T,pre_fac, E_A)

def inv_sum_invs(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return 1/sum(1/arr)


def np_unpacker(var, df, target, scale):
    if 'log' in scale.lower():
        var = 10 ** var

    try:
        df[target] = var
    except ValueError:
        df[target] = [var]

    time = df['time'].to_numpy()[0]
    depth = df['depth'].to_numpy()[0]
    thick = df['thick'].to_numpy()[0]
    diff = df['diff'].to_numpy()[0]
    temp = df['temp'].to_numpy()[0]
    efield = df['efield'].to_numpy()[0]

    return time, depth, thick, diff, temp, efield


def char_time(var, df, target='time', scale='lin'):
    time, depth, thick, diff, temp, efield = np_unpacker(var, df, target, scale)

    mob = diff / (gf.KB_EV * temp)

    return ((2*np.sqrt(diff * time)) + mob * efield * time) - depth


def np_ratio(var, df, target='time', scale='lin'):  # thick,temp,efield,time,diff,mob):
    time, depth, thick, diff, temp, efield = np_unpacker(var, df, target, scale)

    mob = diff / (gf.KB_EV * temp)

    term_A1 = erfc((depth - mob * efield * time) / (2 * np.sqrt(diff * time)))
    term_A2 = erfc(-(depth - 2 * thick + mob * efield * time) / (2 * np.sqrt(diff * time)))
    term_B = erfc(-mob * efield * time / (2 * np.sqrt(diff * time)))
    return (1 / (2 * term_B)) * (term_A1 + term_A2)


def np_cost(var, df, ratio=0.08, target='time', scale='lin'):

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
    def __init__(self, df, depth=None):
        self.df = df.copy()
        if depth is not None:
            self.df['depth'] = depth

    def char_eq(self, target='time', bound=0.5, xmax=1e12):
        """Return sum of squared errors (pred vs actual)."""

        x0 = optimize.root_scalar(char_time, bracket=(0, xmax), args=(self.df.copy(), target),method='bisect').root
        bounds = (x0*(1-bound),x0*(1+bound))

        # return {"x0": np.log10(x0), "bounds": np.log10(bounds)}
        return {"x0": x0, "bounds": bounds}

    def test_range(self):
        times = np.logspace(1, 12, 100)
        ratio_array_full = np_ratio(times, self.df.copy(), 'time', 'log')

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

        return {"x0": np.log10(x0), "bounds": np.log10(bounds)}


    def np_sim(self, pred='char', topC=None, botC=None, ls_params={}, bound=0.5, xmax=1e12):
        """Return sum of squared errors (pred vs actual)."""

        ls_params = {**{"jac": "3-point", "xtol": 1e-12}, **ls_params}

        vals = self.char_eq('time', bound, xmax)

        val = {**vals, **ls_params}

        try:
            results = up.Time(
                optimize.least_squares(
                    np_cost,
                    args=(
                        self.df.copy(),
                        botC / topC,
                        'time',
                        'lin',
                    ),
                    **val,
                ).x[0]
            )
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
        material='mat',
        data_guide={"resis": 1, "perm": 1, "diff": 1, "thick": 1},
        temp=25,
        efield=0,
        volt=0,
        area=1,
        topC=1e21,
        botC=1e10,
        btt_solver=False,
        ls_params={},
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

        self.btt = up.Time(1)

        if btt_solver:
            self.btt_solver(ls_params)

        self.rt_link=rt_link
        self.dt_link=dt_link

    @property
    def thick(self):
        """Return thickness of layer."""
        return up.Length(self.data.var1["thick"], self.data.var2["thick"])

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
                self.data.var1["diff"] = val
                self.data.var2["diff"] = "cm"

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
            self.data.var2["resis"] / (gf.KB_EV * self.temp.K)
        )

    @resistivity.setter
    def resistivity(self, val):
        """ sets rho_0 to val """
        if val is not None:
            if self.rt_link and self.data.var2["resis"] != 0:
                self.temp = up.Temp(self.data.var2["resis"] / (gf.KB_EV * np.log(val/self.data.var1["resis"])), 'K')
            else:
                self.data.var1["resis"] = val
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
        if not hasattr(self, '_er'):
            self._er = self.data.var1["perm"]

        return self._er

    @er.setter
    def er(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            val = gf.Sig_figs(val,2)
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
            self.er = val * self.thick.cm / (self.area.cm * up.Length(gf.PERM, "m", -1).cm)

    @property
    def charge(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.er * up.Length(gf.PERM, "m", -1).cm * self.resistivity * self.curr.A

    @charge.setter
    def charge(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            self.er = val / (up.Length(gf.PERM, "m", -1).cm * self.resistivity * self.curr.A)

    @property
    def diff(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.data.var1["diff"] * np.exp(
            -self.data.var2["diff"] / (gf.KB_EV * self.temp.K)
        )

    @diff.setter
    def diff(self, val):
        if val is not None:
            if self.dt_link and self.data.var2["diff"] != 0:
                self.temp = up.Temp(self.data.var2["diff"] / (gf.KB_EV * np.log(val/self.data.var1["diff"])), 'K')
            else:
                self.data.var1["diff"] = val
                self.data.var2["diff"] = 0


    @property
    def mob(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.diff / (gf.KB_EV * self.temp.K)

    @property
    def btt(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_btt'):
            self._btt = up.Time(24, 'hr', 1, 's')
        return self._btt

    @btt.setter
    def btt(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if val is not None:
            if isinstance(val, up.Time):
                self._btt = val
            elif isinstance(val, (tuple, list)):
                self._btt = up.Volt(*val)
            elif isinstance(val, (dict)):
                self._btt = up.Volt(**val)
            else:
                self._btt = up.Volt(val, "V")

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
            "BTT": float(self.btt.class_values()[
                np.log10(self.btt.class_values()) >= 0
            ].min()),
            "BTT unit": str(self.btt.class_values()[
                np.log10(self.btt.class_values()) >= 0
            ].idxmin()),
        }

        return pd.DataFrame([info])

    @property
    def np_info(self):
        """Return the necessary information."""

        df = {
            "time": float(self.btt.s),
            "depth": float(self.thick.cm),
            "thick": float(self.thick.cm),
            "diff": float(self.diff),
            "temp": float(self.temp.K),
            "efield": float(self.efield.V),
            "top conc": float(self.topC),
            "bot conc": float(self.botC),
            "BTT": float(self.btt.s),
        }

        return pd.DataFrame([df])

    def btt_solver(self, pred='char', topC=None, botC=None, depth=None, ls_params={}, bound=0.5):
        """Return sum of squared errors (pred vs actual)."""
        if self.diff != 0:
            if topC is not None:
                self.topC = topC
            if botC is not None:
                self.botC = botC
            btt_np = NernstPlanck(self.np_info, depth)
            self.btt = btt_np.np_sim(pred, topC, botC, ls_params, bound)
        return self.btt.s

    def data_imp(self, material=None, data_guide=None):
        if material is not None:
            self.material = material
        if data_guide is not None:
            self.data_guide = data_guide
        self.data = MatDatabase().get_df(self.material, **self.data_guide)

class Stack:
    def __init__(
        self,
        layers=[["boro",{}], ["eva",{}], ["sinx",{}]],
        **layer_kwargs,
    ):

        if isinstance(layers, (int, np.integer)):
            self._layers = [Layer() for _ in range(layers)]
        elif all(isinstance(lay, Layer) for lay in layers):
            self.layers = layers
        else:
            self.layers = [Layer(material=mat, data_guide=info, **layer_kwargs) for mat, info in layers]

    @property
    def uniform_vars(self):
        return ['temp','curr','area']

    @property
    def stack_info(self):
        """Return sum of squared errors (pred vs actual)."""
        self._stack_info = pd.concat([lay.info for lay in self.layers], axis=0, ignore_index=True)
        return self._stack_info

    @property
    def np_stack(self):
        """Return sum of squared errors (pred vs actual)."""
        self._np_stack = pd.concat([lay.np_info for lay in self.layers], axis=0, ignore_index=True)
        return self._np_stack

    @property
    def thick(self):
        """Return thickness of layer."""
        return up.Length(np.array([lay.thick.cm for lay in self.layers]), 'cm')

    @thick.setter
    def thick(self, val):
        if isinstance(val, up.BaseUnits):
            self.set_attrs('thick', list(val))
        else:
            self.set_attrs('thick', val)

    @property
    def temp(self):
        """Return thickness of layer."""
        return up.Temp(np.array([lay.temp.K for lay in self.layers]).mean(), 'K')

    @temp.setter
    def temp(self, val):
        if isinstance(val, up.BaseUnits):
            self.set_attrs('temp', list(val))
        else:
            self.set_attrs('temp', val)

    @property
    def area(self):
        """ Return Area """
        return up.Length(np.array([lay.area.cm for lay in self.layers]).mean(), 'cm')

    @area.setter
    def area(self, val):
        if isinstance(val, up.BaseUnits):
            self.set_attrs('area', list(val))
        else:
            self.set_attrs('area', val)

    @property
    def efield(self):
        """ Return V/cm as volt type asuming cm's """
        return up.Volt(np.array([lay.efield.V for lay in self.layers]), 'V')

    @efield.setter
    def efield(self, val):
        """Ensures input is a Volt class"""
        if isinstance(val, up.BaseUnits):
            self.set_attrs('efield', list(val))
        else:
            self.set_attrs('efield', val)

    @property
    def resistivity(self):
        """ rho = rho0 * exp(Ea / k_B T) """
        return np.array([lay.resistivity for lay in self.layers])

    @resistivity.setter
    def resistivity(self, val):
        """ sets rho_0 to val """
        if isinstance(val, up.Res):
            self.set_attrs('resistivity', val.Ohm)
        elif isinstance(val, up.Length):
            self.set_attrs('resistivity', val.cm)
        else:
            self.set_attrs('resistivity', val)

    @property
    def volt(self):
        """ V = E * t """
        return up.Volt(np.array([lay.volt.V for lay in self.layers]), 'V')

    @volt.setter
    def volt(self, val):
        """ E = V / t """
        if isinstance(val, up.BaseUnits):
            self.set_attrs('volt', list(val))
        else:
            self.set_attrs('volt', val)

    @property
    def resistance(self):
        """ R = rho * t / A """
        return up.Res(np.array([lay.resistance.Ohm for lay in self.layers]), 'Ohm')

    @resistance.setter
    def resistance(self, val):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(val, up.BaseUnits):
            self.set_attrs('resistance', list(val))
        else:
            self.set_attrs('resistance', val)

    @property
    def curr(self):
        """Return sum of squared errors (pred vs actual)."""
        return up.Curr(np.array([lay.curr.A for lay in self.layers]).mean(), 'A')

    @property
    def er(self):
        """Return sum of squared errors (pred vs actual)."""
        return np.array([lay.er for lay in self.layers])

    @er.setter
    def er(self, val):
        """Return sum of squared errors (pred vs actual)."""
        self.set_attrs('er', val)

    @property
    def cap(self):
        """Return sum of squared errors (pred vs actual)."""
        return np.array([lay.cap for lay in self.layers])

    @property
    def charge(self):
        """Return sum of squared errors (pred vs actual)."""
        return np.array([lay.charge for lay in self.layers])

    @property
    def diff(self):
        """Return sum of squared errors (pred vs actual)."""
        return np.array([lay.diff for lay in self.layers])

    @diff.setter
    def diff(self, val):
        self.set_attrs('er', val)

    @property
    def mob(self):
        """Return sum of squared errors (pred vs actual)."""
        return np.array([lay.mob for lay in self.layers])

    def set_attrs(self, param='volt', array=0):
        if isinstance(array, (int, float, np.integer, np.float)):
            """if single value, generates all from that single value """
            array = np.ones(len(self.layers)) * array
        elif not isinstance(array[0], (int, float, np.integer, np.float)):
            """if 1st value isnt a number that means that array is a list of unit array"""
            tmp = [array[1:] for _ in range(len(self.layers))]
            if param in self.uniform_vars:
                [tmp[n].insert(0,array[0].max()) for n in range(len(self.layers))]
            else:
                [tmp[n].insert(0,array[0][n]) for n in range(len(self.layers))]
            array = tmp
        elif len(array) != len(self.layers) or isinstance(array[1], str):
            """if 1st value isnt a number that means that array is a list of unit array"""
            array = [array for _ in range(len(self.layers))]
        else:
            """should only be arrays left,"""
            if param in self.uniform_vars:
                array = [array.max() for _ in range(len(self.layers))]


        [setattr(self.layers[m], param, array[m]) for m in range(len(self.layers))]

class VoltDev(object):

    def __init__(self, stack, sys_volt=0, **kwargs):

        if not isinstance(stack, Stack):
            stack = Stack(stack, **kwargs)

        self.stack = stack

        self.sys_volt = sys_volt

    @property
    def area(self):
        return self.stack.area

    @area.setter
    def area(self, val):
        if val != self.stack.area.cm and val != self.stack.area:
            self.stack.area = val

    @property
    def sys_res(self):
        return up.Res(self.stack.resistance.Ohm.sum())

    @property
    def curr(self):
        return up.Curr(self.stack.curr.A.ave())

    @property
    def sys_volt(self):
        if not hasattr(self, '_sys_volt'):
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

        self.stack.volt = self.stack.resistance.Ohm / self.sys_res.Ohm * self._sys_volt.V
        # [setattr(self.layers[n], 'volt', volts[n]) for n in range(len(volts))]

    @property
    def sys_cap(self):
        """Return sum of squared errors (pred vs actual)."""
        return inv_sum_invs(self.stack.cap)

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
        return (sum(self.stack.charge)+self.sys_charge)/(len(self.stack.layers)+1)

    def bulk_param(self, params):
        """Update all layer information from dict"""
        try:
            [[self.single_param(n, key, val[n]) for n in range(len(self.stack.layers))] for key, val in params.items()]
            # [[setattr(self.stackl.layers[n], key, val[n]) for n in range(len(self.stack.layers))] for key, val in params.items()]
        except (AttributeError, IndexError):
            print("Format Error: Must be in form {param:[x1,x2,...,xn],...}, where n is is length of list and uniform")

    def single_param(self, layer_ind=0, param='volt', val=0, copy=False):
        """Update single layer information  and find needed system volt to accomodate this"""
        stack_copy = copy(self.stack)
        sys_volt_copy = self.sys_volt
        layer = self.stack.layers[layer_ind]

        if hasattr(layer, param):
            setattr(layer, param, val)

        self.sys_volt = layer.volt.V * self.sys_res.Ohm / layer.resistance.Ohm

        if copy:
            result = {'eff_volt': copy(self.sys_volt), 'stack': copy(self.stack)}
            self.stack = stack_copy
            self.sys_volt = sys_volt_copy
            return result
        else:
            return

    def curr_eval(self, current=0, lay_params={}, alter_volt=False):
        """Calculate needed R to explain current and related eff_volt"""
        if isinstance(current, up.Curr):
            tar_curr = current
        else:
            tar_curr = up.Curr(current)

        if lay_params == {}:
            lay_params= {'material':'air','temp': self.stack.layers[0].temp, 'area':self.stack.layers[0].area}

        added_layer = Layer(**lay_params)
        added_layer.volt = tar_curr.A * added_layer.resistance.Ohm

        eff_volt = up.Volt(self.sys_volt.V - added_layer.volt.V)

        if alter_volt:
            self.sys_volt = eff_volt

        return {'eff_volt':eff_volt, 'layer':added_layer}

    def cap_eval(self, dampener=0.05, iters=100, qtol=0.05, ):

        n = 0
        dev=1
        while dev > qtol and n < iters:
            eps0 = np.array(self.stack.er)
            # eps1 = (self.sys_cap * self.sys_res.Ohm) / (up.Length(gf.PERM, "m", -1).cm * self.stack.resistivity.Ohm)
            eps1 = self.sys_charge / (up.Length(gf.PERM, "m", -1).cm * self.stack.efield.V  * self.area.cm)
            eps_dev = eps0 - eps1
            epsf = eps0 - eps_dev * dampener
            # [setattr(self.layers[m], 'er', epsf[m]) for m in range(len(epsf))]

            setattr(self.stack.layers[abs(eps_dev/eps0).argmax()], 'er', epsf[abs(eps_dev/eps0).argmax()])

            dev = max(abs(self.ave_charge - np.array(self.stack.charge)) / self.ave_charge)
            n+=1

        return

class Module(object):

    def __init__(self, stack, sys_volt=0, size=100, dep_var={}, ind_var='', **kwargs):

        if not isinstance(stack, Stack):
            stack = Stack(stack, **kwargs)

        self.stack = stack

        self.sys_volt = sys_volt

        self.size = size
        self.units = {'volt': up.Volt,
                      'sys_volt': up.Volt,
                      'curr': up.Curr,
                      'thick': up.Length,
                      'area': up.Length,
                      'efield': up.Volt,
                      'resistivity': up.Res,
                      'resistance': up.Res,
                      'sys_res': up.Res,
                      }


    @property
    def ax_std(self):
        """ log scales: efield """

        return {'temp':['lin',20,100],
                'efield':['log',3,6],
                }

    @property
    def ax_list(self):
        """ log scales: efield """
        return self.stack.area

    @ax_list.setter
    def ax_list(self, val):

        if val != self.stack.area.cm and val != self.stack.area:
            self.stack.area = val

    @property
    def xaxis(self):
        return self.stack.area

    # @xaxis.setter
    # def xaxis(self, val):d
    #     if val != self.stack.area.cm and val != self.stack.area:
    #         self.stack.area = val

    def ax_generator(self, param, min_val, max_val, unit):
        if np.log10(max_val/min_val) > 1:
            scale = 'log'
        else:
            scale = 'lin'

        if param.lower() in ['area','thick']:1

        if scale == 'log':
            array = np.logspace(min_val,max_val,self.size)


        return array






#%%  BTT general sim

# layer_names = ['soda','eva','sinx']

mod_root = [
    ['soda', {"resis": 2, "thick": "metric"}],
    ['eva', {"resis": 1, "diff": "max", "thick": "thin"}],
    ['sinx', {"resis": 1}],
]

test_temp = 60
test_volt = 1000

lay_init = [Layer(material=mat, data_guide=info, temp=test_temp, area=1) for mat, info in mod_root]

lay_active = [copy(lay) for lay in lay_init]

stack1 = Stack(lay_active)

circuit = VoltDev(stack1, sys_volt=test_volt)
circuit.cap_eval()

# circuit = VoltDev(lay_active, sys_volt=test_volt)

# volt_test.sys_volt = 1500
# mod_info=circuit.stack

# air_layer = circuit.curr_eval(1.6e-9)

# np_test = NernstPlanck(lay_active[1].np_info)
# var = np_test.char_eq()
# test = np_test.np_sim(pred='char', topC=1e21, botC=1e10)

# var2 = np_test.char_eq(target='diff')
#%%


# circuit.cap_eval()

# mod_info.iloc[:,1:-1] = mod_info.iloc[:,1:-1].astype(float)


# mod_info=pd.concat([lay.info for lay in layers], axis=0,ignore_index=True)
#%%

# mod_eva1 = Module(layers=["soda", "eva", "sinx"], temp=test_temp, volt=test_volt)
# mod_eva1.thickness_adj(
#     layer=[0, 1, 2], thick=[gf.tocm(3.2, "mm"), gf.tocm(450, "um"), gf.tocm(80, "nm")]
# )
# res_eva1 = mod_eva1.module

# mod_eva2 = Module(layers=["soda", "eva_alt", "sinx"], temp=test_temp, volt=test_volt)
# mod_eva2.thickness_adj(
#     layer=[0, 1, 2], thick=[gf.tocm(3.2, "mm"), gf.tocm(450, "um"), gf.tocm(80, "nm")]
# )
# res_eva2 = mod_eva2.module

# mod_poe1 = Module(layers=["soda", "poe_a", "sinx"], temp=test_temp, volt=test_volt)
# mod_poe1.thickness_adj(
#     layer=[0, 1, 2], thick=[gf.tocm(3.2, "mm"), gf.tocm(450, "um"), gf.tocm(80, "nm")]
# )
# res_poe1 = mod_poe1.module

# mod_poe2 = Module(layers=["soda", "poe_b", "sinx"], temp=test_temp, volt=test_volt)
# mod_poe2.thickness_adj(
#     layer=[0, 1, 2], thick=[gf.tocm(3.2, "mm"), gf.tocm(450, "um"), gf.tocm(80, "nm")]
# )
# res_poe2 = mod_poe2.module

# mod_poe3 = Module(layers=["soda", "poe_c", "sinx"], temp=test_temp, volt=test_volt)
# mod_poe3.thickness_adj(
#     layer=[0, 1, 2], thick=[gf.tocm(3.2, "mm"), gf.tocm(450, "um"), gf.tocm(80, "nm")]
# )
# res_poe3 = mod_poe3.module
