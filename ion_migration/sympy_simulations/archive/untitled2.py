# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:08:19 2023

@author: j2cle
"""

import sympy as sp
from utilities import Length, Q_E

x = sp.Symbol('x[0]')
f, g = sp.symbols('f g', cls=sp.Function)

er = 2.65
c_surf=1e18
c_base=1e-20
x_i = Length(10, "nm").um

pfunc = sp.Piecewise((-c_surf*Q_E/er, x <= x_i), (-c_base*Q_E/er, x > x_i))
# pfunc2 = sp.Piecewise((-c_surf/er, x <= x_i), (-c_base/er, x > x_i))

diffeq = sp.Eq(f(x).diff(x,x), pfunc)
test = sp.Eq(f(x).diff(x,x), 0)

pres1 = sp.integrate(pfunc, x)
pres2 = sp.integrate(pres1, x)
var1 = sp.printing.ccode(pres2)
res1 = sp.dsolve(diffeq, f(x))
var2  = sp.printing.ccode(res1)
