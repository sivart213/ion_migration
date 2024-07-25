#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 00:12:33 2022

@author: jake
"""
import os

import logging
import pandas as pd
import numpy as np
from pathlib import Path

import ion_migration.fem_simulations as pnpfs
from research_tools.functions import convert_val, scatter, find_path, pathlib_mk

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
    tempC = 80,
    voltage = 1500,
    thick = convert_val(450, "um", "cm"),  # 450 um
    time_s = convert_val(1, "day", "s"),
    diffusivity =  4e-16,  # 1.5e-15 # nominal=4e-16
    er = 2.95,  # 2.65, updated
    material = "EVA",
    csurf = 1e12, # Surface Conc atoms/cm2 csurf_vol * thick_na.cm # Surface Conc atoms/cm2
    # cbulk = 1e-20,
    )

mesh = dict(
    voltage_pin=True,
    # thick_na = convert_val(5, "nm", "cm"),  # 200 need 0.4 for 5e19
    # thick_mesh = thick * 1.00,  # Percent of thickness
    thick_mesh_ref = convert_val(25, "um", "cm"),  # 5 um
    dx_max = convert_val(10, "nm", "cm"),  # 10 nm
    dx_min = convert_val(0.01, "nm", "cm"),  # 0.01 nm
    dt_max = convert_val(3, "h", "s"),  # 10 min
    # dt_base = 3,
    )

other = dict(
    valence = 1,
    csurf_vol = 2.24e+22,  # atoms/cm3 in Na layer Na in pure NaCl = 2.2422244077479557e+22
    # screen = 0.33,
    )


# {"diffusivity": 0.75e-15, "valence": 0.5, "screen": 0.33, "rate": 1e-8, "m_eq": 2.24e20}
runs = {

        # "EVA_PNPCL_01": {"screen": None},
        # "EVA_PNPCL_02": {"screen": None, "valence2": 1},
        # "EVA_PNPCL_03": {"screen": None, "valence2": -1},
        # "EVA_PNPCL_04": {"screen": None, "valence2": 1, "valence3": -1},
        # "EVA_PNPCL_05": {"screen": None, "valence2": -1, "valence3": 1},
        
        "EVA_PNPCL_06": {"screen": "calc"},
        "EVA_PNPCL_07": {"screen": "calc", "valence2": 1},
        "EVA_PNPCL_08": {"screen": "calc", "valence2": -1},
        "EVA_PNPCL_09": {"screen": "calc", "valence2": 1, "valence3": -1},
        "EVA_PNPCL_10": {"screen": "calc", "valence2": -1, "valence3": 1},
        
        "EVA_PNPCL_11": {"screen": "volt"},
        "EVA_PNPCL_12": {"screen": "volt", "valence2": 1},
        "EVA_PNPCL_13": {"screen": "volt", "valence2": -1},
        "EVA_PNPCL_14": {"screen": "volt", "valence2": 1, "valence3": -1},
        "EVA_PNPCL_15": {"screen": "volt", "valence2": -1, "valence3": 1},
        
        # for 9
        # "EVA_22": {"screen": 0.33, "rate": 5e-9},
        # "EVA_23": {"screen": 0.33, "rate": 5e-11},
        # "EVA_24": {"screen": 0.33, "rate": 5e-15},
        
        # for 10
        # "EVA_22": {"screen": "volt", "rate": 5e-9},
        # "EVA_23": {"screen": "volt", "rate": 5e-11},
        # "EVA_24": {"screen": "volt", "rate": 5e-15},
        
        # for 11
        # "EVA_mPNP_23": {"screen": "calc", "rate": 5e-9},
        # "EVA_mPNP_24": {"screen": "calc", "rate": 5e-11},
        # "EVA_mPNP_25": {"screen": "calc", "rate": 5e-15},
        
        # for 12
        # "EVA_PNP_30": {"screen": None, "rate": 5e-9},
        # "EVA_PNP_31": {"screen": None, "rate": 5e-11},
        # "EVA_PNP_32": {"screen": None, "rate": 5e-15},
        
        }


rpath = find_path("Work Docs","Data", "Raw", "Simulations", "PNP", "mPNPCL_80_r01",  base=find_path("ASU Dropbox", base="drive")) # "mPNP_80_r9"  "mPNP_80_r10"  "mPNP_80_r11"   "mPNP_80_r12"
pathlib_mk(rpath)
gnote = "Cl at x=0" #"compare calc mPNP and PNP"

for kw, var in runs.items():
    note = gnote + ", ".join([f"{k}={v}" for k, v in var.items()]) 
    file_tag = kw #alternate 4 & 5
    
    h5FileName = str(rpath / Path(f"{file_tag}.h5"))
    myLogger = getLogger(str(rpath), file_tag, name='SL')

    rate = var.pop("rate",  3e-14) #0.5e-12
    m_eq = var.pop("m_eq",  4.48e19)/var.get("csurf_vol", other["csurf_vol"])


    inputs = dict(
        note = note,
        in_flux = ("interf", [rate, m_eq]), #("box", 0) or  ("interf",[1,1]) ("surf", rate),
        # out_flux = ("closed", 0), # ("surf", rate), or ("box", 0) or ("interf",[1,1])

        max_calls = 1,
        max_iter = 1000,

        fcallLogger = myLogger,
        h5_storage = h5FileName,
        debug = True,
    )

    inputs = {**base, **mesh, **other, **inputs, **var}
    def passed_func(x, c, p):

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

        if xrange == 0:
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
                grid=True,
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
                grid=True,
                )
        # print("Total Na: {:e}".format(integrate.trapz(c, x*1e-4)))

    inputs["func"] = passed_func


    # %%
    
    t_sim, x1i, c1i, p1i, c_max = pnpfs.single_layer(**inputs)
    print("")

    if 0:
        df = pd.DataFrame(np.array([x1i, c1i, p1i]).T, columns = ["depth","conc","voltage"])
        df.plot(x="depth", y="conc", logy=True, grid=True)
        df.plot(x="depth", y="voltage", grid=True)

