# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:00:29 2023

@author: j2cle
"""

import sympy as sp
import numpy as np
from utilities import Length, Q_E, PERMI__CM

f = sp.symbols('f', cls=sp.Function)
x = sp.Symbol('x[0]', real=True)

vapp = 1500
thick_na = Length(50, "um").cm
thick = Length(450, "um")
depth = np.linspace(0, thick.cm, 10000)

er = 2.65
c_surf = 1e18
c_base = 1e-20


conc = sp.Piecewise((0, x < 0),
                    (c_surf, x <= thick_na),
                    # (c_base, x <= thick.cm-thick_na),
                    (c_base, x <= thick.cm),
                    (0, x > thick.cm),
                    evaluate=False)

ion_dens =  sp.Piecewise((0, x < 0),
                    (c_surf * Q_E/(er*PERMI__CM), x <= thick_na),
                    # (c_base * Q_E/(er*PERMI__CM), x <= thick.cm-thick_na),
                    (c_base * Q_E/(er*PERMI__CM), x <= thick.cm),
                    (0, x > thick.cm),
                    evaluate=False)

ode_v = sp.Eq(f(x).diff(x,x), -1*ion_dens)
eion_s = float(sp.integrate(x/thick.cm*conc, (x, 0, thick.cm)) * -1*Q_E/(2.65*PERMI__CM))

# f_eion1 = sp.dsolve(ode_v, f(x), simplify=False, ics={f(thick.cm/2) : 0})
# f_eion2 = sp.dsolve(ode_v, f(x), simplify=False, ics={f(x).diff(x).subs(x, thick.cm/2) : 0})
f_eion3 = sp.dsolve(ode_v, f(x), simplify=False, ics={f(thick.cm) : 0, f(x).diff(x).subs(x, thick.cm) : eion_s})

var_c=sp.lambdify(x, conc)(depth)
var_e=sp.lambdify(x, f_eion3.rhs.diff(x))(depth)
var_v=sp.lambdify(x, f_eion3.rhs)(depth)
# sp.printing.ccode(f_v)
print("Total Na: {:e}".format(sp.integrate(conc, (x, 0, thick.cm))))
