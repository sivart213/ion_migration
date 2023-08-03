#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 00:12:33 2022

@author: jake
"""
import os
import pnptransport.infinite_sl_JC7 as pnpfs
import logging
import pandas as pd
import numpy as np
from functools import partial
from scipy import integrate
from pathlib import Path

import periodictable as pt
from utilities import Length, Time, BOLTZ__EV, Temp, scatter, p_find, pathlib_mk

def g_to_atoms(element, atoms=None, grams=None):
    if isinstance(element, str):
        chemical = pt.formula(element)
    else:
        chemical = element
    res_dict = chemical.atoms
    res_dict["res"] = 1
    if atoms is not None:
        result = atoms / pt.constants.avogadro_number * chemical.mass
    else:
        result = grams / chemical.mass * pt.constants.avogadro_number
    res_dict = {key: val * result for key, val in res_dict.items()}

    return res_dict


def getLogger(out_path, filetag, **kwargs):
    """
    Gets the logger for the cuurent simulation

    Parameters
    ----------
    out_path: str
        The path to store the log
    filetag: str
        The fle tag of the log file
    kwargs:
        name: str
            The name to use as a prefix

    Returns
    -------

    """
    name = kwargs.get('name', '')
    logFile = os.path.join(out_path, filetag + "_{}.log".format(name))
    logging.getLogger().handlers.clear()
    # get the myLogger
    pnp_logger = logging.getLogger('simlog')
    pnp_logger.setLevel(logging.DEBUG)
    pnp_logger.handlers.clear()

    # create file handler which logs even debug messages
    fh = logging.FileHandler(logFile)
    fh.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to the logger
    pnp_logger.addHandler(fh)
    pnp_logger.addHandler(ch)

    return pnp_logger

base = dict(
    tempC = 60,
    voltage = 1500,
    thick = Length(450, "um").cm,  # 450 um
    time_s = Time(8, "d").s,
    diffusivity = 2e-16,  # 1.5e-15 # nominal
    er = 2.95,  # 2.65, updated
    material = "EVA",
    csurf = 1e8, # Surface Conc atoms/cm2 csurf_vol * thick_na.cm # Surface Conc atoms/cm2
    # cbulk = 1e-20,
    )

mesh = dict(
    voltage_pin=True,
    # thick_na = Length(5, "nm").cm,  # 200 need 0.4 for 5e19
    # thick_mesh = thick * 1.00,  # Percent of thickness
    thick_mesh_ref = Length(25, "um").cm,  # 5 um
    dx_max = Length(10, "nm").cm,  # 10 nm
    dx_min = Length(0.01, "nm").cm,  # 0.01 nm
    dt_max = Time(1, "hr").s,  # 10 min
    # dt_base = 3,
    )

other = dict(
    valence = 0.4,
    csurf_vol = 2.24e+22,  # atoms/cm3 in Na layer Na in pure NaCl = 2.2422244077479557e+22
    )


# {"diffusivity": 0.75e-15, "valence": 0.4, "screen_leng": 1, "rate": 1e-12, "m_eq": 1.12e19}
runs = {
        # "EVA_01": {"diffusivity": 1e-16, "rate": 1e-12, "m_eq": 2.24e18},
        # "EVA_02": {"diffusivity": 1e-16, "rate": 1e-12, "m_eq": 1.12e19},
        # "EVA_03": {"diffusivity": 1e-16, "rate": 1e-12, "m_eq": 2.24e19},
        # "EVA_04": {"diffusivity": 1e-16, "rate": 1e-12, "m_eq": 1.12e20},

        "EVA_05": {"diffusivity": 1e-16, "rate": 1e-14, "m_eq": 2.24e18},
        "EVA_06": {"diffusivity": 1e-16, "rate": 1e-14, "m_eq": 1.12e19},
        "EVA_07": {"diffusivity": 1e-16, "rate": 1e-14, "m_eq": 2.24e19},
        "EVA_08": {"diffusivity": 1e-16, "rate": 1e-14, "m_eq": 1.12e20},

        "EVA_09": {"diffusivity": 1.1e-16, "rate": 1e-12, "m_eq": 2.24e18},
        "EVA_10": {"diffusivity": 1.1e-16, "rate": 1e-12, "m_eq": 1.12e19},
        "EVA_11": {"diffusivity": 1.1e-16, "rate": 1e-12, "m_eq": 2.24e19},
        "EVA_12": {"diffusivity": 1.1e-16, "rate": 1e-12, "m_eq": 1.12e20},

        "EVA_13": {"diffusivity": 1.1e-16, "rate": 1e-14, "m_eq": 2.24e18},
        "EVA_14": {"diffusivity": 1.1e-16, "rate": 1e-14, "m_eq": 1.12e19},
        "EVA_15": {"diffusivity": 1.1e-16, "rate": 1e-14, "m_eq": 2.24e19},
        "EVA_16": {"diffusivity": 1.1e-16, "rate": 1e-14, "m_eq": 1.12e20},
        }

for kw, var in runs.items():
    note = ", ".join([f"{k}={v}" for k, v in var.items()]) # f"Debye of {v1} nm an1rr1d h={v2}"
    file_tag = kw #alternate 4 & 5
    rpath = p_find("Data", "Raw", "Simulations", "PNP", "mPNP_f60_r4_5")
    pathlib_mk(rpath)
    h5FileName = str(rpath / Path(f"{file_tag}.h5"))
    myLogger = getLogger(str(rpath), file_tag, name='SL')

    rate = var.pop("rate",  1e-12) #1e-8
    m_eq = var.pop("m_eq",  1.12e19)/other["csurf_vol"]

    inputs = dict(
        note = note,
        in_flux = ("interf", [rate, m_eq]), #("box", 0) or  ("interf",[1,1]) ("surf", rate),
        # out_flux = ("closed", 0), # ("surf", rate), or ("box", 0) or ("interf",[1,1])
        screen = 1/-Length(var.pop("screen_leng",  2), "nm").cm,

        max_calls = 0,
        max_iter = 500,

        fcallLogger = myLogger,
        h5_storage = h5FileName,
        debug = True,
    )

    inputs = {**base, **mesh, **other, **inputs, **var}
    def passed_func(x, c, p):
        # if all(c < 1e5):
        #     return
        try:
            p_max_n = np.argwhere(p==p.max())[0][0]
            p_min_n = np.argwhere(p<=(2*p[0]-p.max()))
            c_min_n = np.argwhere(c<=c[p_max_n:].max()*1e-5)
            # if pmax at 0 or the x where C goes "low" is beyond the 2x drop in p
            if len(p_min_n) == 0 or abs(p[0]) >= abs(p.max()) or x[c_min_n[0][0]] >= x[p_min_n[0][0]]:
                xrange = c_min_n[0][0]
            else:
                xrange = p_min_n[0][0]
        except IndexError:
            xrange = -1


        scatter(pd.DataFrame(np.array([x, c, p]).T, columns = ["depth","conc","voltage"]),
                x="depth",
                y="conc",
                xscale="linear",
                yscale="log",
                # xlimit=[0, x[np.argwhere(c<1e10)[0]]],
                xlimit=[0,x[xrange]], #list(x[xrange]),
                ylimit=[c[:xrange].min()/10, c[:xrange].max()*10],
                name=f"{file_tag}_conc",
                xname=None,
                yname=None,
                zname=None,
                save=rpath,
                show=False,
                hue=None,
                linewidth=0,
                )

        pborder = (p[:xrange].max()-p[:xrange].min())/4
        pmin = p[:xrange].min() - pborder
        pmax = p[:xrange].max() + pborder
        scatter(pd.DataFrame(np.array([x, c, p]).T, columns = ["depth","conc","voltage"]),
                x="depth",
                y="voltage",
                xscale="linear",
                yscale="linear",
                # xlimit=[0, x[np.argwhere(c<1e10)[0]]],
                xlimit=[0,x[xrange]],
                # ylimit=[p[np.argwhere(c<1e10)[0]]-10,p.max()+10],
                ylimit=[pmin, pmax],
                name=f"{file_tag}_volt",
                xname=None,
                yname=None,
                zname=None,
                save=rpath,
                show=False,
                hue=None,
                linewidth=0,
                )
        # print("Total Na: {:e}".format(integrate.trapz(c, x*1e-4)))

    inputs["func"] = passed_func


    # %%
    t_sim, x1i, c1i, p1i, c_max = pnpfs.single_layer(**inputs)

    df = pd.DataFrame(np.array([x1i, c1i, p1i]).T, columns = ["depth","conc","voltage"])

    if 0:
        df.plot(x="depth", y="conc", logy=True, grid=True)
        df.plot(x="depth", y="voltage", grid=True)
