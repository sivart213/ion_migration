# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:12:19 2023

@author: j2cle
"""

import sympy as sp
import numpy as np
from utilities import Length, Q_E, PERMI__CM

f = sp.symbols('f', cls=sp.Function)
x = sp.Symbol('x[0]', real=True)

vapp = 1500
thick_na = Length(1500, "nm").cm
thick = Length(450, "um")
depth = np.linspace(0, thick.cm, 1000)

er = 2.65
c_surf = 1e17
c_base = 1e-20

ode_ebias = sp.Eq(f(x).diff(x), 0)
ode_vbias = sp.Eq(f(x).diff(x, x), 0)
f_vbias = sp.dsolve(ode_vbias, f(x), simplify=False, ics={f(0): vapp, f(thick.cm): 0})
# vbias = sp.lambdify(x, f_vbias.rhs)


conc_part = [[c_surf, thick_na], [c_base, thick.cm]]
conc_init = [(0, x < 0), (0, x > thick.cm)]
[conc_init.insert(-1, (cp[0], x<= cp[1])) for cp in conc_part]

conc = sp.Piecewise((0, x < 0),
                    (c_surf, x <= thick_na),
                    (c_base, x <= thick.cm),
                    (0, x > thick.cm),
                    evaluate=False)

ion_dens =  sp.Piecewise((0, x < 0),
                    (c_surf * Q_E/(er*PERMI__CM), x <= thick_na),
                    (c_base * Q_E/(er*PERMI__CM), x <= thick.cm),
                    (0, x > thick.cm),
                    evaluate=False)

ode_v = sp.Eq(f(x).diff(x,x), -1*ion_dens)

eion_g = float(sp.integrate((x - thick.cm)/thick.cm*conc, (x, 0, thick.cm)) * Q_E/(er*PERMI__CM))
eion_s = float(sp.integrate(x/thick.cm*conc, (x, 0, thick.cm)) * -1*Q_E/(er*PERMI__CM))
ebias = float(f_vbias.rhs.diff(x))

f_vg = sp.dsolve(ode_v, f(x), simplify=False, ics={f(x).diff(x).subs(x, 0): (ebias - eion_g),
                                                  f(thick.cm): 0})
f_vs = sp.dsolve(ode_v, f(x), simplify=False, ics={f(x).diff(x).subs(x, thick.cm): (ebias + eion_s),
                                                  f(thick.cm): 0})

f_v = sp.dsolve(ode_v, f(x), simplify=False, ics={f(x).diff(x).subs(x, thick.cm): -1*(-1*ebias - eion_g + eion_s),
                                                  f(thick.cm): 0})

var_g=sp.lambdify(x, f_vg.rhs)(depth)
var_s=sp.lambdify(x, f_vs.rhs)(depth)
var=sp.lambdify(x, f_v.rhs)(depth)

# sp.printing.ccode(f_v)
print("Total Na: {:e}".format(sp.integrate(conc, (x, 0, thick.cm))))
