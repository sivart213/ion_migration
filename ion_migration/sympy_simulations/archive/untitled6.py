# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:32:52 2023

@author: j2cle
"""

import sympy as sp
import numpy as np
import pandas as pd
from utilities import Length, Q_E, PERMI__CM

f = sp.symbols('f', cls=sp.Function)
x = sp.Symbol('x[0]', real=True)

vapp = 1500
thick_na = Length(50, "um").cm
thick = Length(450, "um")
depth = np.linspace(0, thick.cm, 1000)

er = 2.65
c_surf = 1e1
c_base = 1e-20

ode_ebias = sp.Eq(f(x).diff(x), 0)
ode_vbias = sp.Eq(f(x).diff(x, x), 0)
f_vbias = sp.dsolve(ode_vbias, f(x), simplify=False, ics={f(0): vapp, f(thick.cm): 0})
vbias = sp.lambdify(x, f_vbias.rhs)

conc = sp.Piecewise((0, x < 0),
                    (c_surf, x <= thick_na),
                    (c_base, x <= thick.cm),
                    (0, x > thick.cm),
                    evaluate=False)

ion_dens = conc * Q_E/(er*PERMI__CM)
ode_eion = sp.Eq(f(x).diff(x), ion_dens)
f_eion = sp.dsolve(ode_eion, f(x), simplify=False, ics={f(thick_na/2): 0})

ode_v = sp.Eq(f(x).diff(x), -1*(-1*f_vbias.rhs.diff(x) + f_eion.rhs)) #Grad V = -1*(-1*Vbias + Eion)
ode_v1 = sp.Eq(f(x).diff(x), -1*(-1*f_vbias.rhs.diff(x) + f_eion.rhs)) #Grad V = -1*(-1*Vbias + Eion)
                                                  # f(x).diff(x).subs(x, thick.cm): -1*(-1*f_vbias.rhs.diff(x) + eion_s),

eion_g = sp.integrate((x - thick.cm)/thick.cm*conc, (x, 0, thick.cm)) * Q_E/(er*PERMI__CM)
eion_s = sp.integrate(x/thick.cm*conc, (x, 0, thick.cm)) * -1*Q_E/(er*PERMI__CM)

f_v2 = sp.dsolve(ode_v, f(x), simplify=False, ics={f(x).diff(x).subs(x, 0): float(-1*(-1*f_vbias.rhs.diff(x) - eion_g))})

                                                  # f(x).diff(x).subs(x, thick.cm): -1*(-1*f_vbias.rhs.diff(x) + eion_s),

f_v = sp.dsolve(ode_v, f(x), simplify=False, ics={f(thick.cm): 0})
# f_vg = sp.dsolve(ode_v, f(x), simplify=False, ics={f(x).diff(x).subs(x, 0): -1*(-1*f_vbias.rhs.diff(x) - eion_g)})
# f_vg = sp.dsolve(ode_v, f(x),simplify=False)
# f_vg.rhs = f_vg.rhs.subs("C1", )

#
# ode_vion = sp.Eq(f(x).diff(x,x), -1*ion_dens)
#
# f_vion = sp.dsolve(ode_vion, f(x), simplify=False, ics={f(x).diff(x).subs(x, thick_na/2) : 0})
# f_vion = sp.dsolve(ode_vion, f(x), simplify=False, ics={f(x).diff(x).subs(x, 0) : ,
#                                                         f(x).diff(x).subs(x, thick.cm) : ,
#                                                         })
# eion = sp.lambdify(x, f_eion.rhs)
# vion = sp.lambdify(x, f_vion.rhs)
var_bias=sp.lambdify(x, f_vbias.rhs)(depth)
var=sp.lambdify(x, f_v.rhs)(depth)



# df = pd.DataFrame(np.array([np.linspace(0, thick.um, len(depth)), abs(sp.lambdify(x, ion_dens)(depth)), vbias(depth), vion(depth)]).T, columns=["depth", "ions","vbias", "vion"])
# df.plot(x="depth", y="vbias", grid=True)
# df.plot(x="depth", y="vion", grid=True)
# df.plot(x="depth", y="ions", logy=True, grid=True)
