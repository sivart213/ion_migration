# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:18:42 2023

@author: j2cle
"""

import sympy as sp
import numpy as np
import pandas as pd
from utilities import Length, Q_E, PERMI__CM, eqn, eqn_sets
from utilities import np_ratio, Time, Length, Temp, PERMI__CM, Q_E, BOLTZ__EV

def poisson(**kwargs):
    """Calculate the poisson equation.
    Vars: thick, volt, valence, rel_perm, perm_const, conc
    Var 'perm_const' default in units of F/cm
    'efield' can be passed in lieu of volt"""
    # func_vars = {**{"efield": None, "perm_const": PERMI__CM}, **kwargs}


    thick = kwargs.get("thick", 1)
    volt = kwargs.get("volt", kwargs.get("efield", 1500/thick) * thick)
    valence = kwargs.get("valence", 1)
    rel_perm = kwargs.get("rel_perm", 1)
    perm_const = kwargs.get("perm_const", PERMI__CM)
    conc = kwargs.get("conc", 1)


    # if volt is None and func_vars["efield"] is not None:
    #     volt = func_vars["efield"] * thick

    if "conc" not in kwargs.keys():  # Solve for concentration
        return 2 * rel_perm * perm_const * volt / (Q_E * valence * thick**2)
    elif "volt" not in kwargs.keys() and "efield" not in kwargs.keys(): # Solve for voltage (or Efield)
        return Q_E * valence * conc * thick**2 / (2 * rel_perm * perm_const)
    else: # Solve for the unkown
        params = {
            "volt": volt,
            "valence": valence,
            "perm": perm_const,
            "rel_perm": rel_perm,
            "thick": thick,
            "conc": conc,
            "elec": Q_E,
        }
        target = [key for key in ["valence","rel_perm","thick"] if key not in kwargs.keys()]
        if len(target) > 1:
            return
        eqn = "2*rel_perm*perm*volt/(elec*valence*thick**2)-conc"
        return eqn_sets(params, target=target[0], eqns=eqn, as_set=False)



f, g = sp.symbols('f g', cls=sp.Function)
x = sp.Symbol('x', real=True)
c, q, er, e0, thick, z, V, t = sp.symbols('c q er e0 L z V t', real=True, constant=True)

bias = 3
# pois1 = sp.Eq(f(x).diff(x, x), 0)

conc = sp.Eq(f(x).diff(x, x), -q*z*c/(er*e0))

pois1 = sp.Eq(f(x).diff(x, x), -q*z*c/(er*e0))
bias = sp.Eq(f(x).diff(x, x), 0)
eqs = [pois1, bias]
# pois1 = sp.Eq(V, -q*z*c/(er*e0))
pois_set = sp.solveset(pois1, f(x), domain=sp.Interval(0,sp.oo))
pois_solved = sp.dsolve(pois1, f(x), simplify=False, ics={f(0): bias, f(thick): 0})
# pois_set.contains()
