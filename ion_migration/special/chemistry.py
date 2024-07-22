# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 19:15:02 2023

@author: j2cle
"""

# import logging
# import numpy as np
# import pandas as pd
# import sympy as sp
# import sympy.physics.units as su

import periodictable as pt



def g_to_atoms(element, atoms=None, grams=None):
    chemical = pt.formula(element)
    res_dict = chemical.atoms
    res_dict["res"] = 1
    if atoms is not None:
        result = atoms / pt.constants.avogadro_number * chemical.mass
    else:
        result = grams / chemical.mass * pt.constants.avogadro_number
    res_dict = {key: val * result for key, val in res_dict.items()}

    return res_dict

def cm3_to_ppyw(element, base, cm3, ppb=False):
    element = pt.formula(element)
    base = pt.formula(base)
    mult = 1e-6
    if ppb:
        mult = 1e-9
    conv = base.density / element.mass  * pt.constants.avogadro_number * mult
    return cm3 / conv

def at_to_wt(element, base, at):
    element = pt.formula(element)
    base = pt.formula(base)
    return element.mass / base.mass * at

# def at_to_wt2(element, base, at):
#     element = pt.formula(element)
#     base = pt.formula(base)
#     return element.mass / base.mass * at

def wt_to_at(element, base, wt):
    element = pt.formula(element)
    base = pt.formula(base)
    return base.mass / element.mass * wt

# def cm3_to_pp(element, base, val, ppb=False):
#     element = pt.formula(element)
#     base = pt.formula(base)
#     mult = 1e-6
#     if ppb:
#         mult = 1e-9
#     conv = base.density / element.mass  * pt.constants.avogadro_number * mult
#     return cm3 / conv

def num_to_mass_density(element, val):
    element = pt.formula(element)
    # base = pt.formula(base)
    return val * element.mass / pt.constants.avogadro_number

def mass_fraction(element, base, val):
    element = pt.formula(element)
    # base = pt.formula(base)
    return val * element.mass / pt.constants.avogadro_number

# %% Testing
if __name__ == "__main__":
    # examples
    # x=g_to_atoms("Na")
    Na = pt.formula('Na[23]{+}')
    Cl = pt.formula('Cl{-}')
    NaCl = pt.formula('NaCl', natural_density=2.176)
    NaOH = pt.formula('NaOH', natural_density=2.13)
    H2O = pt.formula('H2O', natural_density=0.9998)
    EVA = pt.formula('28%wt C[12]4H[1]6O[16]2 //  C[12]2H[1]4', natural_density=0.92)

    # Na_mol_mass = g_to_atoms('Na[23]',1e16)["res"] / Na.density
    # H2O_mol_mass = g_to_atoms('H2O',grams=2e-3)["res"] / H2O.density

    # Na_EVA_wt = Na.molecular_mass / EVA.molecular_mass
    # H2O_EVA_wt = H2O.molecular_mass / EVA.molecular_mass

    Si = pt.formula('Si')
    P = pt.formula('P')
    
    
    P_Si_wt = cm3_to_ppyw(P, Si, 5e15)
    P0 = num_to_mass_density(P, 5e16)
    P_weight = P0/(P0+Si.density)
    
    P1 = pt.formula('P', density=.5)
    P2 = pt.formula('P', natural_density=.2)
    P3 = pt.formula('P', natural_density=.3)
    
