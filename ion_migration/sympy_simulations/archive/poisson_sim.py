# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:18:42 2023

@author: j2cle
"""

import sympy as sp
import numpy as np
import pandas as pd
from utilities import Length, Q_E, PERMI__CM, eqn
from utilities import np_ratio, poisson, Time, Length, Temp, PERMI__CM, Q_E, BOLTZ__EV

f, g, h = sp.symbols('f g h', cls=sp.Function)
x = sp.Symbol('x', real=True)
# deb_leng = sp.symbols('k_0', real=True, constant=True)
c, c_b, thick_na, volt, deb_leng = sp.symbols('c c_b L_n V k_0', real=True, constant=True, nonnegative=True) # includes 0
q, er, valence, thick_bulk = sp.symbols('q e_r z L_b', real=True, constant=True, positive=True) # does not include 0

# l = sp.symbols('L', real=True, constant=True, positive=True)



vals_b = {thick_na: Length(0.5,"um").cm,
        q: Q_E,
        er: 2.95*PERMI__CM,
        deb_leng: 1/Length(1, "nm").cm,
        valence: 0.4,
        }
vals_v = {volt: 1500}
vals_c = {c: 2e19,
          c_b: 1e-20,
          }
vals_a = {**vals_b, **vals_c, **vals_v}

depth = np.linspace(-Length(0.6,"um").cm, Length(0.6,"um").cm, 1000)

perm = er*sp.exp(x*deb_leng)

thick = thick_na + thick_bulk


cent_ion =  sp.Piecewise((0, x < -1*thick_na/2),
                    (-1*c*q*valence/perm, x <= thick_na/2),
                    (0, x > thick_na/2),
                    evaluate=False)

orig_ion =  sp.Piecewise((0, x < 0),
                    (-1*c*q*valence/perm, x <= thick_na),
                    (0, x > thick_na),
                    evaluate=False)


eion = sp.integrate(-cent_ion, (x))
vion = sp.integrate(eion, (x))


eion_0c = sp.integrate(x/thick_na*cent_ion, (x, -thick_na/2, thick_na/2))
eion_0c2 = 2*sp.integrate(x/thick_na*cent_ion, (x, 0, thick_na/2))
eion_0 = sp.integrate(x/thick_na*orig_ion, (x, 0, thick_na))

vion_0 = sp.integrate(x*orig_ion, (x, 0, thick_na))


eion_totc = sp.integrate(cent_ion/thick_na, (x, -thick_na/2, thick_na/2)).args[0][0]
eion_totc2 = sp.integrate(cent_ion/thick_na, (x, 0, thick_na/2))
eion_tot = sp.integrate(orig_ion/thick_na, (x, 0, thick_na))

x_barc = eion_0c/eion_totc
x_bar = eion_0/eion_tot


# fcent1 = sp.dsolve(sp.Eq(f(x).diff(x,x), cent_ion), simplify=False, ics={f(x_barc):0, f(x).diff(x).subs(x, x_barc): 0}).rhs
# fcent2 = sp.dsolve(sp.Eq(f(x).diff(x,x), cent_ion), simplify=False, ics={f(x_barc.subs(vals_a)):0, f(x).diff(x).subs(x, x_barc.subs(vals_a)): 0}).rhs
# fcent3 = sp.dsolve(sp.Eq(f(x).diff(x,x), cent_ion), simplify=False, ics={f(x_barc):0, f(x).diff(x).subs(x, thick_na/2): x_barc/thick_na*eion_totc2}).rhs
# fcent4 = sp.dsolve(sp.Eq(f(x).diff(x,x), cent_ion), simplify=False, ics={f(x_barc.subs(vals_a)):0, f(x).diff(x).subs(x, thick_na.subs(vals_a)/2): (x_barc/thick_na*eion_0c2).subs(vals_a)}).rhs
# fcent5 = sp.dsolve(sp.Eq(f(x).diff(x,x), cent_ion), simplify=False, ics={f(x_barc):0, f(x).diff(x).subs(x, thick_na/2): x_barc/thick_na*eion_tot}).rhs
# fcent6 = sp.dsolve(sp.Eq(f(x).diff(x,x), cent_ion), simplify=False, ics={f(x_barc.subs(vals_a)):0, f(x).diff(x).subs(x, thick_na.subs(vals_a)/2): (x_barc/thick_na*eion_0c).subs(vals_a)}).rhs


forig1 = sp.dsolve(sp.Eq(f(x).diff(x,x), orig_ion), simplify=False, ics={f(x_bar):0, f(x).diff(x).subs(x, x_bar): 0}).rhs
# forig2 = sp.dsolve(sp.Eq(f(x).diff(x,x), orig_ion), simplify=False, ics={f(x_bar.subs(vals_a)):0, f(x).diff(x).subs(x, x_bar.subs(vals_a)): 0}).rhs
forig3 = sp.dsolve(sp.Eq(f(x).diff(x,x), orig_ion), simplify=False, ics={f(0):0, f(x).diff(x).subs(x, thick_na): x_bar*eion_tot}).rhs
# forig4 = sp.dsolve(sp.Eq(f(x).diff(x,x), orig_ion), simplify=False, ics={f(x_bar.subs(vals_a)):0, f(x).diff(x).subs(x, thick_na.subs(vals_a)): (x_bar/thick_na*eion_tot).subs(vals_a)}).rhs
# forig5 = sp.dsolve(sp.Eq(f(x).diff(x,x), orig_ion), simplify=False, ics={f(x).diff(x).subs(x, x_bar): 0, f(x).diff(x).subs(x, thick_na): x_bar/thick_na*eion_tot}).rhs
# forig6 = sp.dsolve(sp.Eq(f(x).diff(x,x), orig_ion), simplify=False, ics={f(x).diff(x).subs(x, x_bar.subs(vals_a)): 0, f(x).diff(x).subs(x, thick_na.subs(vals_a)): (x_bar/thick_na*eion_tot).subs(vals_a)}).rhs
results = {
    # "Symb, Cent @ 0": fcent1,
    # "Cent @ 0": fcent2,
    # "Symb, Cent @ half": fcent3,
    # "Cent @ half": fcent4,
    # "Symb, Cent @ end": fcent5,
    # "Cent @ end": fcent6,
    "Symb, Orig @ 0": forig1,
    # "Orig @ 0": forig2,
    "Symb, Orig @ end": forig3,
    # "Orig @ end": forig4,
    # "Symb, Orig @ 0-end": forig5,
    # "Orig @ 0-end": forig6,
    }

for key, res in results.items():
    resv = sp.lambdify(x, res.subs(vals_a))(depth)
    rese = -1*sp.lambdify(x, res.subs(vals_a).diff(x))(depth)
    pd.DataFrame(dict(x=depth*1e4, y=resv), columns=["x","y"]).plot(x="x",y="y", title=f"V: {key}", grid=True)
    pd.DataFrame(dict(x=depth*1e4, y=rese), columns=["x","y"]).plot(x="x",y="y", title=f"E: {key}", grid=True)
