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

def layer_btt(
    time, depth, temp, e_app, diff, mob, target=0.08, thick=450e-4
):
    time = 10**time
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
        elif isinstance(val,(tuple,list)):
            self._thick = up.Length(*val)
        elif isinstance(val,(dict)):
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
        elif isinstance(val,(tuple,list)):
            self._temp = up.Temp(*val)
        elif isinstance(val,(dict)):
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
        elif isinstance(val,(tuple,list)):
            self._area = up.Length(*val)
        elif isinstance(val,(dict)):
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
        elif isinstance(val,(tuple,list)):
            self._volt = up.Volt(*val)
        elif isinstance(val,(dict)):
            self._volt = up.Volt(**val)
        else:
            self._volt = up.Volt(val, "V")

    @property
    def resistivity(self) -> float:
        """Return sum of squared errors (pred vs actual)."""
        return  self._res_pre * np.exp(self._res_ea / (gf.KB_EV * self._temp.K))

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
        if not hasattr(self,"_btt"):
            self.btt = {'jac':'3-point','xtol':1e-12}
        times = np.logspace(1,12,100)
        a3 = c_ratio_np(self.thick, self.thick, self._temp.K, self.efield, times, self.diff, self.mob)
        a33 = np.where((a3 > 0) & (a3 < 1))[0]
        if len(a33) == 0:
            x0=times[a3[::-1].argmin()]
            bounds=(times[a3[::-1].argmin()],times[a3.argmax()])
        elif len(a33) == 1:
            x0=times[a33]
            bounds=(times[a33-1][0],times[a33+1][0])
        elif len(a33) == 2:
            x0=times[a33[0]]
            bounds=(times[a33[0]][0],times[a33[-1]][0])
        else:
            x0=times[a33[0]]
            bounds=(times[a33[0]][0],times[a33[-1]][0])

        vals={'x0':np.log10(x0),'bounds':np.log10(bounds)}
        val={**vals, **self._btt_vals}
        btt = 10**optimize.least_squares(layer_btt,args=(self.thick, self._temp.K, self.efield, self.diff, self.mob, self.botC/self.topC, self.thick), **val).x
        self._btt = up.Time(btt)
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


# @dataclass
# class Layer2(object):
#     """Return sum of squared errors (pred vs actual)."""

#     # Invisible attribute (init-only)
#     material: str = "undefined"
#     thick: float = gf.tocm(500, "um")
#     diff_type: InitVar[str] = "undefined"

#     # Initialized attribute
#     temp: float = 25
#     efield: float = 0
#     resistivity: float = field(init=False)

#     # Generated attribute
#     diff: float = field(init=False)
#     mob: float = field(init=False)

#     area: float = 1
#     res: float = field(init=False)
#     volt: float = field(init=False)
#     cap: float = field(init=False)
#     charge: float = field(init=False)

#     @property
#     def resistivity(self) -> float:
#         """Return sum of squared errors (pred vs actual)."""
#         return self._res_pre * np.exp(self._res_ea / (gf.KB_EV * gf.CtoK(self.temp)))

#     @resistivity.setter
#     def resistivity(self, _):
#         pass

#     @property
#     def diff(self) -> float:
#         """Return sum of squared errors (pred vs actual)."""
#         return self._d_0 * np.exp(-self._e_a / (gf.KB_EV * gf.CtoK(self.temp)))

#     @diff.setter
#     def diff(self, _):
#         pass

#     @property
#     def mob(self) -> float:
#         """Return sum of squared errors (pred vs actual)."""
#         return self.diff / (gf.KB_EV * gf.CtoK(self.temp))

#     @mob.setter
#     def mob(self, _):
#         pass

#     @property
#     def volt(self) -> float:
#         """Return sum of squared errors (pred vs actual)."""
#         return self.efield * self.thick

#     @volt.setter
#     def volt(self, _):
#         pass

#     @property
#     def res(self) -> float:
#         """Return sum of squared errors (pred vs actual)."""
#         return self.resistivity * self.thick / self.area

#     @res.setter
#     def res(self, _):
#         pass

#     @property
#     def cap(self) -> float:
#         """Return sum of squared errors (pred vs actual)."""
#         return self._er * self.area / self.thick

#     @cap.setter
#     def cap(self, _):
#         pass

#     @property
#     def charge(self) -> float:
#         """Return sum of squared errors (pred vs actual)."""
#         return self.cap * self.volt

#     @charge.setter
#     def charge(self, _):
#         pass

#     def __post_init__(self, diff_type):
#         """Return layer object."""

#         self._diff_type = diff_type.lower()
#         try:
#             self._res_pre = gf.mat_database.loc[self.material.lower(), "pre"]
#         except KeyError:
#             self._res_pre = gf.mat_database.loc["eva", "pre"]
#         try:
#             self._res_ea = gf.mat_database.loc[self.material.lower(), "ea"]
#         except KeyError:
#             self._res_ea = gf.mat_database.loc["eva", "ea"]
#         try:
#             self._er = gf.mat_database.loc[self.material.lower(), "perm"]
#         except KeyError:
#             self._er = gf.mat_database.loc["eva", "perm"]
#         try:
#             self._d_0 = gf.diff_arrh.loc[diff_type.lower(), "pre"]
#         except KeyError:
#             self._d_0 = gf.diff_arrh.loc["ave", "pre"]
#         try:
#             self._e_a = gf.diff_arrh.loc[diff_type.lower(), "ea"]
#         except KeyError:
#             self._e_a = gf.diff_arrh.loc["ave", "ea"]

class Module:
    def __init__(self, layers=["boro", "eva", "sinx"], temp=25, array=100, focus="eva"):

        self.layers = layers
        self.layer_list = [Layer(material=layer, temp=temp) for layer in layers]
        self.sys_temp = temp

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
        # info_df['time'] = 1
        # info_df['low'] = 0.0
        # info_df['high'] = 0.0
        # info_df['changed'] = True
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


test = Layer(temp=85,efield=100000.0)

times = np.logspace(1,12,100)
a3=c_ratio_np(test.thick, test.thick, test._temp.K, test.efield, times, test.diff,test.mob)
a33 = np.where((a3 > 0) & (a3 < 1))[0]
if len(a33) == 0:
    x0=times[a3[::-1].argmin()]
    bounds=(times[a3[::-1].argmin()],times[a3.argmax()])
elif len(a33) == 1:
    x0=times[a33]
    bounds=(times[a33-1][0],times[a33+1][0])
elif len(a33) == 2:
    x0=times[a33[0]]
    bounds=(times[a33[0]][0],times[a33[-1]][0])
else:
    x0=times[a33[0]]
    bounds=(times[a33[0]][0],times[a33[-1]][0])

val={'x0':np.log10(x0),'bounds':np.log10(bounds),'jac':'3-point','xtol':1e-12}
a1=optimize.least_squares(layer_btt,args=(test.thick,test._temp.K,test.efield,test.diff,test.mob,0.08,test.thick),**val)


# mod_soda=Module()

# # mod_boro=Module()
# # mod_boro.glass = 'Boro'
# # mod_boro.resistivity()

# # mod_soda.bbt('run1',layer='EVA', target=1e10, source=5e21, diff_range='Ave')


# # plotcont6(mod_soda.T_1D,mod_soda.E_1D,mod_soda.bbt_arr['run1'],mod_soda.rho_layer_Tdep,mod_boro.rho_layer_Tdep,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])
# # plotcont8(mod_soda.T_1D,mod_soda.E_1D,mod_soda.bbt_arr['run1'],mod_soda.rho_layer_Tdep,mod_boro.rho_layer_Tdep,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])

# # #%% simulation of stress testing
# # # start regular module
# mod_soda=Module(layers=['soda','eva','sinx'], temp=85)
# mod_soda.thickness_adj(layer=[0,1,2],thick=[gf.tocm(3.2,'mm'), gf.tocm(450,'um'),gf.tocm(80,'nm')])
# mod_soda.bbt('run1', layer='EVA', target=1e10, source=5e21, diff_range='max')
# # # mod_soda.thickness_adj()
# # # E in EVA for 1500 V and 80 C
# mod_soda.sys_volt = 1000
# info_soda = mod_soda.module
# field_E = mod_soda.focus_df['efield']

# # # generate test layout
# # eset_mod = Module(layers=['soda','eva','soda'], temp=60)
# # # eset_mod.rho_sinx = 3.3e16
# # # eset_mod.sinx_t = gf.tocm(50,'um')
# # eset_mod.thickness_adj(layer=[0,1,2],thick=[gf.tocm(2,'mm'), gf.tocm(450,'um'),gf.tocm(2,'mm')])
# # eset_mod.focus_list.efield = field_E
# # best_v = eset_mod.ext_volt
# # eset_mod.sys_volt = 1500

# # eset_curr = eset_mod.leakage(4)
# # info_eset = eset_mod.module
# # find V input


# # find V input
# # test_V = eset_mod.find_ext_vdm(temp=80,field=field_E)

# # #%% simulation of stress testing
# # # start regular module
# # mod_soda=Module()
# # # mod_soda.thickness_adj()
# # # E in EVA for 1500 V and 80 C
# # field_E = mod_soda.find_layer_vdm(temp=60,volt=1500,full_calcs=False)

# # # generate test layout
# # mod_soda_mims2 = Module()
# # mod_soda_mims2.rho_sinx = 3.3e16
# # # eset_mod.sinx_t = gf.tocm(50,'um')
# # mod_soda_mims2.thickness_adj(module=0,glass=gf.tocm(4,'mm'),sinx=gf.tocm(100,'um'))
# # # find V input


# # #%% 3
# # eset_mod.bbt(ident='run1',target=1e17,source=5e19,diff_range='max',diff_depth=gf.tocm(1,'um'))

# # # #%% 4
# # # test_time = eset_mod.find_time('run1',80,test_V)
# # # test_depths = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)

# # #%% 5
# # eset_mod.bbt(ident='run2',target=1e17,source=5e19,diff_range='min',diff_depth=gf.tocm(2,'um'))

# # # test_time = eset_mod.find_time('run2',80,test_V)
# # # test_depths = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)

# # #%% 6
# # eset_mod.bbt(ident='run3',target=1e17,source=5e19,diff_range='ave',diff_depth=gf.tocm(2,'um'))

# # # test_time = eset_mod.find_time('run3',80,test_V)
# # # test_depths = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)


# # #%% 7
# # # test_depths_alt = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time,temp=80,volt=test_V)

# # #%% 8 set simulation for current diffuision
# # eset_mod.bbt(ident='run1',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(2,'um'))
# # eset_mod.bbt(ident='run2',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(1,'um'))
# # eset_mod.bbt(ident='run3',target=1e17,source=5e19,diff_range='min',diff_depth=gf.tocm(1,'um'))

# # mod_soda_mims2.bbt(ident='run1',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(2,'um'))
# # mod_soda_mims2.bbt(ident='run2',target=1e17,source=5e19,diff_range='tof',diff_depth=gf.tocm(1,'um'))
# # mod_soda_mims2.bbt(ident='run3',target=1e17,source=5e19,diff_range='min',diff_depth=gf.tocm(1,'um'))
# # #%% 9
# # test_time_60 = eset_mod.find_time('run3',60,1500)
# # test_depths_60 = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time_60,temp=80,volt=1500)
# # test_diffs_60 = eset_mod.diff_range

# # test_time_80 = mod_soda_mims2.find_time('run3',80,1500)
# # test_depths_80_air = mod_soda_mims2.find_depth_range(target=1e17,source=5e19,time=test_time_80,temp=80,volt=1500)
# # test_diffs_80_air = mod_soda_mims2.diff_range


# # #%% 10
# # test_time_60 = eset_mod.find_time('run1',80,test_V)
# # test_depths_60 = eset_mod.find_depth_range(target=1e17,source=5e19,time=test_time_60,temp=60,volt=test_V)


# # #%% 10

# # eset_mod.leakage(2**2,80,1500)
# # I_80=eset_mod.I_leakage_Tdep[gf.find_nearest(eset_mod.T_1D,80+273.15)]
# # V_meas_80 = I_80*1.5e6
# # V_meas_80_alt = I_80*680e3


# # #%% 11
# # mod_boro=Module()
# # mod_boro.enc_pre=mod_boro.glass_pre
# # mod_boro.enc_ea=mod_boro.glass_ea
# # mod_boro.arrhenius_data('boro')
# # mod_boro.rho_sinx=4e12
# # all_glass=gf.tocm(.125,'in')
# # mod_boro.thickness_adj(module=0,glass=gf.tocm(.125,'in'),enc=gf.tocm(2,'mm'),sinx=gf.tocm(100,'um'))

# # mod_boro.leakage(2**2,80,test_V)
# # I_80_rev=mod_boro.I_leakage_Tdep[gf.find_nearest(mod_boro.T_1D,80+273.15)]
# # V_meas_80_rev = I_80*1.5e6
# # V_meas_80_alt_rev = I_80*680e3


# # #%% 12
# # test_time_75 = eset_mod.find_time('run4',75,test_V)
# # test_time_70 = eset_mod.find_time('run4',70,test_V)
# # test_time_65 = eset_mod.find_time('run4',65,test_V)
# # test_time_60 = eset_mod.find_time('run4',60,test_V)

# # #%% 13


# # mod_low_sinx=Module()
# # mod_low_sinx.rho_sinx = 1e10
# # mod_low_sinx.resistivity(layer_focus='sinx')
# # mod_low_sinx.find_layer_vdm(80,1500)

# # print(mod_low_sinx.E_layer)

# # mod_high_sinx=Module()
# # mod_high_sinx.rho_sinx = 1e15
# # mod_high_sinx.resistivity(layer_focus='sinx')
# # mod_high_sinx.find_layer_vdm(80,1500)

# # print(mod_high_sinx.E_layer)
