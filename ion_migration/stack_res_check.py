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
"""

# %% Imports
import numpy as np
import pandas as pd

import research_tools.equations as jce
import research_tools.functions as jcf

# %% Code


# %% Operations
if __name__ == "__main__":
    T = 80 + 273.15
    A = 1
    V = 1500

    # glass = (0.0450, -0.9703)  # Boro-Duran
    glass = (0.0382, -0.8842)  # Soda-Vidrasa

    R = dict(
        air=jce.resistance(
            1e14,
            A,
            jcf.convert_val(1, "um", "cm"),
        ),
        glass1=jce.resistance(
            jce.arrh(glass[0], glass[1], T), A, jcf.convert_val(2, "mm", "cm")
        ),
        eva=jce.resistance(
            jce.arrh(29430, -0.6, T), A, jcf.convert_val(450, "um", "cm")
        ),
        glass2=jce.resistance(
            jce.arrh(glass[0], glass[1], T), A, jcf.convert_val(2, "mm", "cm")
        ),
    )

    res = jce.voltage_divider(R, V, "eva")

    print("Vdrop = {0:.4E}".format(res))
