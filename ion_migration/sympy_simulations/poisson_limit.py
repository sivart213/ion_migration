# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:18:42 2023

@author: j2cle
"""

import sympy as sp
import numpy as np
import pandas as pd
import sympy.physics.units as su

from defect_code.equations import screened_permitivity
from defect_code.functions import get_const, map_plt

def get_lim(eqs, sol1, sol2, vals={}, multiplier=-1):
    if isinstance(eqs, sp.Piecewise):
        vlim_eq = [sp.Eq(sol1, multiplier * arg[0]) for arg in eqs.args]
        clim_eq = [sp.solveset(f, sol2) for f in vlim_eq]

        res = {f"volt_eq{n+1}": vlim_eq[n].rhs for n in range(len(vlim_eq))}
        res = {**res, **{f"conc_eq{n+1}": clim_eq[n].args[0] for n in range(len(clim_eq))}}
        try:
            if vals == {}:
                raise TypeError
            res = {**res, **{f"volt_lim{n+1}": float(vlim_eq[n].rhs.subs(vals)) for n in range(len(vlim_eq))}}
            res = {**res, **{f"conc_lim{n+1}": float(clim_eq[n].args[0].subs(vals)) for n in range(len(clim_eq))}}

            res = {**res, **{f"volt_{n+1}": (str(res[f"volt_eq{n+1}"]), res[f"volt_lim{n+1}"]) for n in range(len(vlim_eq))}}
            res = {**res, **{f"conc_{n+1}": (str(res[f"conc_eq{n+1}"]), res[f"conc_lim{n+1}"]) for n in range(len(clim_eq))}}
        except TypeError:
            res = {**res, **{f"volt_{n+1}": str(res[f"volt_eq{n+1}"]) for n in range(len(vlim_eq))}}
            res = {**res, **{f"conc_{n+1}": str(res[f"conc_eq{n+1}"]) for n in range(len(clim_eq))}}

    elif isinstance(eqs, sp.Mul):
        vlim_eq = sp.Eq(sol1, multiplier * eqs)
        clim_eq = sp.solveset(vlim_eq, sol2)
        try:
            if vals == {}:
                raise TypeError
            res = {
                "volt_eq1": vlim_eq.rhs,
                "conc_eq1": clim_eq.args[0],
                "volt_lim1": float(vlim_eq.rhs.subs(vals)),
                "conc_lim1": float(clim_eq.args[0].subs(vals)),
                "volt_1": (str(vlim_eq.rhs), float(vlim_eq.rhs.subs(vals))),
                "conc_1": (str(clim_eq.args[0]), float(clim_eq.args[0].subs(vals))),
                }
        except TypeError:
            res = {
                "volt_eq1": vlim_eq.rhs,
                "conc_eq1": clim_eq.args[0],
                "volt_1": str(vlim_eq.rhs),
                "conc_1": str(clim_eq.args[0]),
                }
    return res


f = sp.symbols("f", cls=sp.Function)
x = sp.Symbol("x", real=True)
# includes 0
C, volt, deb_leng = sp.symbols("C_Na V k_0", real=True, constant=True, nonnegative=True)

# does not include 0
q, er, valence, thick = sp.symbols("q epsilon z L", real=True, constant=True, positive=True)

# Constants
e0 = get_const("e0", False, [su.farad, su.cm])
k_B = get_const("boltzmann",False, [su.eV, su.K])
um2cm = float(su.convert_to(su.um, su.cm).n().args[0])

vals = {
    thick: 0.5 * um2cm,
    q: get_const("elementary_charge",False,  [su.C]),
    er: 2.95 * e0,
    volt: 1500,
    C: 2e19,
    deb_leng: 1 / float(su.convert_to(100*su.nm, su.cm).n().args[0]),
    valence: 1,
}



perm = er * sp.exp(x * deb_leng * 0)
# perm = screened_permitivity(er, deb_leng, x)

orig_ion = sp.Piecewise(
    (0, x < 0), (C * q * valence / perm, x <= thick), (0, x > thick), evaluate=False
)


eion = sp.integrate(orig_ion, (x))
vion = sp.integrate(-eion, (x))

eion2 = sp.integrate(orig_ion, (x, 0, thick))
vion2 = sp.integrate(-eion2, (x, 0, thick))

vion_0 = sp.integrate(x * -1 * orig_ion, (x, 0, thick))

# res_vals = get_lim(vion_0, volt, C, vals={**vals})["conc_eq1"]
# res_dict =  get_lim(vion_0, volt, C)

# res_eq = res_dict["volt_eq1"]

# depth = np.linspace(0, float(su.convert_to(5*su.um, su.cm).n().args[0]), 500)
# concs = np.logspace(15, 20, 500)
# conc_res = np.ndarray((0, 3))

# for con in concs:
#     res1 = sp.lambdify(thick, res_eq.subs({**{k: v for k, v in vals.items() if k != thick}, **{C: con}}))(depth)
#     conc_res = np.vstack((conc_res, np.stack((depth*1e4, np.full_like(depth, con), res1),1)))

# conc_res[:,2][~np.isfinite(conc_res[:,2])] = np.max(conc_res[:,2][np.isfinite(conc_res[:,2])])
# map_plt(depth*1e4, concs, np.resize(conc_res[:,2], (500,500)), "linear", "log", "log", ztick=10)

# if perm != er:
#     res_eq = res_dict["conc_eq1"]
#     base_eq = res_dict["conc_eq2"]

#     depth = np.linspace(float(su.convert_to(1*su.nm, su.cm).n().args[0]),float(su.convert_to(5*su.um, su.cm).n().args[0]), 500)
#     screens = np.linspace(float(su.convert_to(1*su.nm, su.cm).n().args[0]), float(su.convert_to(100*su.nm, su.cm).n().args[0]), 500)
#     # screens = np.logspace(-7, -5, 500)
#     screen_res = np.ndarray((0,7))
#     vals_sim = {k: v for k, v in vals.items() if k != thick}
#     for scr in screens:
#         res1 = sp.lambdify(thick, base_eq.subs({**vals_sim, **{deb_leng: 1 / scr}}))(depth)
#         res2 = sp.lambdify(thick, res_eq.subs({**vals_sim, **{deb_leng: 1 / scr}}))(depth)
#         res3 = sp.lambdify(thick, res_eq.subs({**vals_sim, **{deb_leng: 1 / scr}, **{valence: 0.75}}))(depth)
#         res4 = sp.lambdify(thick, res_eq.subs({**vals_sim, **{deb_leng: 1 / scr}, **{valence: 0.5}}))(depth)
#         res5 = sp.lambdify(thick, res_eq.subs({**vals_sim, **{deb_leng: 1 / scr}, **{valence: 0.25}}))(depth)
#         screen_res = np.vstack((screen_res, np.stack((depth*1e4, np.full_like(depth, scr), res1, res2, res3, res4, res5),1)))
#     for n in range(2,7):
#         screen_res[:,n][~np.isfinite(screen_res[:,n])] = np.max(screen_res[:,n][np.isfinite(screen_res[:,n])])
