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
    # logging.basicConfig(filename=logFile, level=logging.INFO)
    # get the myLogger
    pnp_logger = logging.getLogger('simlog')
    pnp_logger.setLevel(logging.DEBUG)
    pnp_logger.handlers.clear()
    # if not pnp_logger.hasHandlers():
    # if pnp_logger.handlers[0] != logging.getLogger().handlers[0]:

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


# {"diff": 0.75, "valence": 0.5, "screen_leng": 1, "rate": 1e-8, "m_eq": 2.24e20}

runs = {
        "EVA_01": {"diff": 3, "valence": 0.4, "rate": 1e-10},
        "EVA_02": {"diff": 3, "valence": 0.4, "rate": 1e-10, "m_eq": 1.15e19},
        "EVA_03": {"diff": 3, "valence": 0.4, "rate": 1e-12},
        "EVA_04": {"diff": 3, "valence": 0.4, "rate": 1e-12, "m_eq": 1.12e19},
        "EVA_05": {"diff": 2, "valence": 0.4, "rate": 1e-10},
        "EVA_06": {"diff": 2, "valence": 0.4, "rate": 1e-10, "m_eq": 1.12e19},
        "EVA_07": {"diff": 2, "valence": 0.4, "rate": 1e-12},
        "EVA_08": {"diff": 2, "valence": 0.4, "rate": 1e-12, "m_eq": 1.12e19},
        # "EVA_09": {"diff": 1.25, "valence": 0.4, "rate": 1e-12, "screen_leng": 2},
        # "EVA_10": {"diff": 1.25, "valence": 0.4, "rate": 1e-13, "m_eq": 1.12e19, "screen_leng": 2},
        # "EVA_11": {"diff": 1.25, "valence": 0.4, "rate": 1e-12, "screen_leng": 2},
        # "EVA_12": {"diff": 1.25, "valence": 0.4, "rate": 1e-13, "m_eq": 1.12e19, "screen_leng": 2},
        }         
for kw, var in runs.items():
    thick = Length(450, "um")  # 450 um
    # thick_na = Length(5, "nm")  # 200 need 0.4 for 5e19

    # thick_mesh = thick * 1.00  # Percent of thickness
    thick_mesh_ref = Length(25, "um")  # 5 um
    dx_max = Length(50, "nm")  # 10 nm
    dx_min = Length(0.001, "nm")  # 0.01 nm
    thick_na = dx_min*2 # Length(1, "nm")  # 200 need 0.4 for 5e19

    t_total = Time(8, "d").s #4 gets odd 5 gets evens
    dt_max = Time(5, "hr")  # 10 min

    volt = 1500
    temp = 60
    diff = var.get("diff", 5)*1e-16  # 1.5e-15 # nominal
    valence = var.get("valence", 0.5)
    perm = 2.95  # 2.65, updated
    
    screen_leng = 1/-Length(var.get("screen_leng", 1), "nm").cm

    csurf_vol = 2.24e+22  # atoms/cm3 in Na layer Na in pure NaCl = 2.2422244077479557e+22
    c_surf = 1e8 # Surface Conc atoms/cm2 csurf_vol * thick_na.cm # Surface Conc atoms/cm2
    rate = var.get("rate", 1e-8) #1e-8
    m_eq = var.get("m_eq", 2.24e20)/csurf_vol 

    def passed_func(x, c, p):
        # if all(c < 1e5):
        #     return

        try:
        # if pmax at 0 or the x where C goes "low" is beyond the 2x drop in p
            p_max_n = np.argwhere(p==p.max())[0][0]
            p_min_n = np.argwhere(p<=(2*p[0]-p.max()))[0][0]
            c_min_n = np.argwhere(c<=c[p_max_n:].max()*1e-5)[0][0]
            if abs(p[0]) >= abs(p.max()) or x[c_min_n] >= x[p_min_n]:
                xrange = c_min_n
            else:
                xrange = p_min_n
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

    note = ", ".join([f"{k}={v}" for k, v in var.items()]) # f"Debye of {v1} nm an1rr1d h={v2}"
    file_tag = kw #alternate 4 & 5
    rpath = p_find("Data", "Raw", "Simulations", "PNP", "mPNP_f60_r3")
    pathlib_mk(rpath)
    h5FileName = str(rpath / Path(f"{file_tag}.h5"))
    myLogger = getLogger(str(rpath), file_tag, name='SL')
    

    inputs = dict(
        material = "EVA",
        note = note,

        time_s = t_total, #345600.0,
        tempC = temp,
        voltage = volt,
        voltage_pin=True,

        thick = thick.cm, #thick,
        # thick_mesh = thick_mesh.cm,
        thick_mesh_ref = thick_mesh_ref.cm,
        # thick_na = thick_na.cm,

        csurf = c_surf, #1.5E+19,
        csurf_vol = csurf_vol, #1.5E+19,
        # cbulk = 1e-20,
        diffusivity = diff, #1.5e-15,
        er = perm,
        valency = valence,
        screen = screen_leng,
        in_flux = ("interf", [rate, m_eq]), #("box", 0) or  ("interf",[1,1]) ("surf", rate),
        # out_flux = ("closed", 0), # ("surf", rate), or ("box", 0) or ("interf",[1,1])

        # xpoints = x_points,
        dx_max=dx_max.cm,
        dx_min=dx_min.cm,
        # xrefine = nor,
        # refine_perc = dor,
        dt_max=dt_max.s,
        # tpoints = t_points, #0.03124972873499362
        # dt_base = 3,
        max_calls = 1,
        max_iter = 500,

        fcallLogger = myLogger,
        h5_storage = h5FileName,
        debug = True,

        func=passed_func,
    )

    # %%
    t_sim, x1i, c1i, p1i, c_max = pnpfs.single_layer(**inputs)

    df = pd.DataFrame(np.array([x1i, c1i, p1i]).T, columns = ["depth","conc","voltage"])

    if 0:
        df.plot(x="depth", y="conc", logy=True, grid=True)
        df.plot(x="depth", y="voltage", grid=True)
