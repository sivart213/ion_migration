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
Created on Tue Dec 19 19:30:10 2023
"""

# %% Imports
import numpy as np
import pandas as pd
import sympy as sp
import sympy.physics.units as su
# from sympy.physics.units.unitsystem import UnitSystem
# from sympy.physics.units.quantities  import Quantity
# from sympy.physics.units.dimensions  import Dimension
from sympy.physics.units import UnitSystem, Quantity, Dimension, DimensionSystem
from sympy.physics.units.systems.si import dimsys_SI, SI
from functools import reduce
# %% Code
def find_unit(quantity, unit_system="SI"):
    """
    Return a list of matching units or dimension names.

    - If ``quantity`` is a string -- units/dimensions containing the string
    `quantity`.
    - If ``quantity`` is a unit or dimension -- units having matching base
    units or dimensions.

    Examples
    ========

    >>> from sympy.physics import units as u
    >>> u.find_unit('charge')
    ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    >>> u.find_unit(u.charge)
    ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    >>> u.find_unit("ampere")
    ['ampere', 'amperes']
    >>> u.find_unit('angstrom')
    ['angstrom', 'angstroms']
    >>> u.find_unit('volt')
    ['volt', 'volts', 'electronvolt', 'electronvolts', 'planck_voltage']
    >>> u.find_unit(u.inch**3)[:9]
    ['L', 'l', 'cL', 'cl', 'dL', 'dl', 'mL', 'ml', 'liter']
    """
    unit_system = UnitSystem.get_unit_system(unit_system)

    import sympy.physics.units as u
    rv = []
    if isinstance(quantity, str):
        rv = [i for i in dir(u) if quantity in i and isinstance(getattr(u, i), Quantity)]
        dim = getattr(u, quantity)
        if isinstance(dim, Dimension):
            rv.extend(find_unit(dim))
    else:
        for i in sorted(dir(u)):
            other = getattr(u, i)
            if not isinstance(other, Quantity):
                continue
            if isinstance(quantity, Quantity):
                if quantity.dimension == other.dimension:
                    rv.append(str(i))
            elif isinstance(quantity, Dimension):
                if other.dimension == quantity:
                    rv.append(str(i))
            elif other.dimension == Dimension(unit_system.get_dimensional_expr(quantity)):
                rv.append(str(i))
    return sorted(set(rv), key=lambda x: (len(x), x))

# %% Operations
if __name__ == "__main__":
    from pathlib import Path
    res_V1 = su.util.quantity_simplify(5*su.joule/su.coulomb, True, "SI")
    res_V2 = su.find_unit(su.voltage, "SI")
    
    si_depend = dict(su.si.dimsys_SI.dimensional_dependencies)
    base_units = {d.dimension: d for d in SI._base_units}
    si_units = su.si.SI.derived_units
    cgs_units = {**si_units, **{
     su.length: su.centimeter,
     su.mass: su.gram,
     su.velocity: su.centimeter/su.second,
     su.acceleration: su.centimeter/su.second**2,
     }}
    
    # test_perm_dims = su.si.SI.get_quantity_dimension(su.vacuum_permittivity)
    list_1 = [cgs_units[su.Dimension(d)] for d in su.vacuum_permittivity.dimension.atoms(sp.Symbol)]
    
    res = su.convert_to(su.vacuum_permittivity, list_1)
    
    
    # var_dim_dict = dimsys_SI.get_dimensional_dependencies(su.vacuum_permittivity.dimension)
    # # var_dim_expr = 1
    # # [var_dim_expr*(k**v) for k, v in var_dim_dict.items()]
    # var_dim_expr = reduce(lambda x, y: x * y, (
    #         si_units[d]**e for d, e in var_dim_dict.items()
    #     ), 1)
    
