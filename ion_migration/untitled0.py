# -*- coding: utf-8 -*-
"""
Insert module description/summary.

Provide any or all of the following:
1. extended summary
2. routine listings/functions/classes
3. see also
4. notes
5. references
6. examples

@author: j2cle
Created on Thu Apr 18 15:37:42 2024
"""

# %% Imports
import numpy as np
import pandas as pd
import sympy as sp
import sympy.physics.units as su
from sympy.physics.units.prefixes import PREFIXES as P
from sympy.physics.units.prefixes import prefix_unit
import research_tools.equations as rte
import research_tools.functions as rtf

# %% Code
def circle_area(r):
    """Calculate the area of a circle."""
    res = np.pi*r**2
    return res

# def parse_prefix(val):
# def mag(x):
#     """
#     >>> [mag(Integer(10)**-i) for i in range(10)]
#     [0, -1, -2, -3, -3, -3, -6, -6, -6, -9]
#     >>> [mag(Integer(10)**i) for i in range(10)]
#     [0, 1, 2, 3, 3, 3, 6, 6, 6, 9]
#     """
#     a = abs(x)
#     # if a < 1000 and 1000*a > 1:
#     #     return int(np.log(a)/np.log(10))
#     if a < 1:
#         -int(np.log(1/a)/np.log(10**3))*3
#     return int(np.log(a)/np.log(10**3))*3

# def get_base_unit(expr, unit_system="SI"):
#     if not isinstance(expr, su.Quantity):
#         return expr
#     if isinstance(unit_system, str):
#         unit_system = getattr(su.systems, "SI")
#     base_unit = {u.dimension: u for u in unit_system.get_units_non_prefixed()}
#     return base_unit[expr.dimension]

# def npre(x, u):
#     i = mag(x)
#     bu = su.util.quantity_simplify(5*su.cm, True, "SI").n()
#     if u != bu:
#         x = su.convert_to(x*u, bu)
#         # if isinstance(unit, (su.Quantity, list)):
#     # if not isinstance(u, list):
#     #     u = [u]
#     # try:
#     #     u = [
#     #         getattr(su, un) if isinstance(un, str) else un for un in u
#     #     ]
#     #     u = u[0]
#     # except (AttributeError, ValueError, TypeError):
#     #     pass
    
#     # fact = 3 if engineering else 1
#     # P = {k:v for k, v in su.prefixes.PREFIXES.items()  if v._exponent % fact == 0 }
#     if i == 0:
#         return x * su.get_unit()
#     for k in P:
#         if P[k].args[2] == i:
#             return x/10**i*prefix_unit(u,{k: P[k]})[0]

# %% Operations
if __name__ == "__main__":
    from pathlib import Path
    a = circle_area(2)
    t = rtf.convert_val(450, "um", "cm")
    high_rho = rte.resistance(1e15, a, t)
    med_rho = rte.resistance(1e13, a, t)
    low_rho = rte.resistance(1e11, a, t)
    print("High end: " + rtf.sci_note(high_rho))
    print("Medium: " + rtf.sci_note(med_rho))
    print("low end: " + rtf.sci_note(low_rho))
    # rtf.parse_unit(10*su.cm)
    # npre(333, su.m)
