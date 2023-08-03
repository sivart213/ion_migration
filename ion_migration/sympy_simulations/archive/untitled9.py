# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:58:20 2023

@author: j2cle
"""

import sympy as sp
import numpy as np
from utilities import Length, Q_E, PERMI__CM

f = sp.symbols('f', cls=sp.Function)
x = sp.Symbol('x[0]', real=True)
c, q, er, e0 = sp.symbols('c q er e0', real=True, constant=True)

vapp = 1500
# thick_na = Length(10, "nm").cm
thick = Length(500, "um")
depth = np.linspace(0, thick.cm, 100)

# er = 2.65
c_surf = 1e18
c_base = 1e-20


conc = sp.Piecewise((0, x < 0),
                    (c, x <= thick.cm),
                    (0, x > thick.cm),
                    evaluate=False)

ion_dens =  sp.Piecewise((0, x < 0),
                    (c * q/(er*e0), x <= thick.cm),
                    (0, x > thick.cm),
                    evaluate=False)

ode_v = sp.Eq(f(x).diff(x,x), -1*ion_dens)
eion_s = float(sp.integrate(x/thick.cm*c_surf, (x, 0, thick.cm)) * -1*Q_E/(2.65*PERMI__CM))
f_ion = sp.dsolve(ode_v, f(x), simplify=False, ics={f(thick.cm/2) : 0, f(x).diff(x).subs(x, thick.cm) : eion_s})

var=sp.lambdify(x, f_ion.rhs.subs([(c,c_surf), (q, Q_E), (er,  2.65), (e0, PERMI__CM)]))(depth)
