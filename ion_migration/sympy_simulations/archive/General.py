# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:44:22 2023

@author: j2cle
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import sympy as sp
import sympy.physics.units as u
from sympy.utilities.autowrap import autowrap


def eqn_sets(params, **kwargs):
    """Calculate. generic discription."""
    results = []
    for key, vals in params.items():
        if isinstance(vals, (np.ndarray, list, tuple)):
            tmp = [eqn({**params, **{key: val}}, **kwargs) for val in vals]
            results.append(tmp)
    if results == []:
        results = eqn(params, **kwargs)
    return results


def eqn(params, target="D", eqns="x-1", as_set=True):
    """Calculate. generic discription."""
    x = sp.Symbol("x", positive=True)
    params[target] = x

    if not isinstance(eqns, str):
        expr = sp.parsing.sympy_parser.parse_expr(eqns.pop(0)[1])
        expr = expr.subs(eqns)
    else:
        expr = sp.parsing.sympy_parser.parse_expr(eqns)
    expr = expr.subs(params)

    try:
        res = sp.solveset(expr, x, domain=sp.S.Reals)
        if isinstance(res, sp.sets.sets.EmptySet):
            res = sp.FiniteSet(*sp.solve(expr))
    except Exception:
        res = sp.FiniteSet(*sp.solve(expr))
    if not as_set:
        res = float(list(res)[0])
    return res


# %% Autowrap

# A, x, y = map(sp.IndexedBase, ['A', 'x', 'y'])
# m, n = sp.symbols('m n', integer=True)
# i = sp.Idx('i', m)
# j = sp.Idx('j', n)
# instruction = sp.Eq(y[i], A[i, j]*x[j])

# matvec = autowrap(instruction, backend="cython")

# %%
C, K = sp.symbols('C K', real=True)
kb = u.boltzmann
kb.dimension  # get the dimension set: energy/temperature
kb.scale_factor  # get the raw value
kb2 = u.convert_to(kb, [u.eV, u.K]).n()

vth = kb2 * 300 * u.K

sp.solveset(sp.Eq(C, K-273.15).subs({K: 350}), C)

sp.solveset((K-273.15-C).subs({K: 350}), C)

expr = sp.parsing.sympy_parser.parse_expr("K-273.15-degC")
sp.solveset(expr.subs("degC", 80), "degC")


# degC = celsius = Celsius = u.Quantity("celsius", abbrev="degC")
# degC.set_global_relative_scale_factor(u.K-273.15, u.K)

volt, res = sp.symbols('V_1 R_1', real=True)
voltu = volt*u.V
resu = res*u.ohm


def parse_unit(expr):
    expr_base = {}
    expr_unit = {}
    for x in expr.args:
        if x.has(u.Quantity):
            expr_base[x] = 1
        else:
            expr_unit[x] = 1
    return expr.subs(expr_base), expr.subs(expr_unit)
