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
from research_tools.functions import convert_val, scatter, p_find, pathlib_mk

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

ratio = 0
c_init = 1e14
cv_init = 2.24e+22

base = dict(
    # tempC = 60,
    tempC = 80,
    voltage = 1500,
    thick = convert_val(450, "um", "cm"),  # 450 um
    # time_s = convert_val(8, "day", "s"),
    time_s = convert_val(1, "day", "s"),
    diffusivity = 4e-16,  # 1.5e-15 # nominal=4e-16
    diffusivity2 = 4e-16,  # 1.5e-15 # nominal=4e-16
    er = 2.95,  # 2.65, updated
    material = "EVA",
    csurf = max(1e-20, c_init * (1.0 - ratio)), # Surface Conc atoms/cm2 csurf_vol * thick_na.cm # Surface Conc atoms/cm2
    # c2surf = 1e-20,
    # cbulk = 1e-20,
    )

base["c2surf"] = max(1e-20, c_init - base["csurf"])

mesh = dict(
    voltage_pin=True,
    # thick_na = convert_val(5, "nm", "cm"),  # 200 need 0.4 for 5e19
    # thick_mesh = thick * 1.00,  # Percent of thickness
    thick_mesh_ref = convert_val(25, "um", "cm"),  # 5 um
    dx_max = convert_val(10, "nm", "cm"),  # 10 nm
    dx_min = convert_val(0.01, "nm", "cm"),  # 0.01 nm
    # dt_max = convert_val(6, "h", "s"),  # 10 min
    dt_max = convert_val(3, "h", "s"),  # 10 min
    # dt_base = 3,
    )


other = dict(
    valence = 1,
    csurf_vol = max(1e-20, cv_init * (1.0 - ratio)),  # atoms/cm3 in Na layer Na in pure NaCl = 2.2422244077479557e+22
    # c2surf_vol = cv_init,
    screen = "calc",
    )

other["c2surf_vol"] = max(1e-20, cv_init - other["csurf_vol"])

eq_c = {"c2surf": c_init, "c2surf_vol": cv_init, "rate2": 1e-12, "diffusivity2": 6e-15}
# {"diffusivity": 0.75e-15, "valence": 0.5, "screen": 0.33, "rate": 1e-8, "m_eq": 2.24e20}
runs = {
        # "EVA_mPNPNP_00": {**eq_c, **{"ratio": 1, "m_eq2": 1.12e19}},
        
        # "EVA_mPNPNP_01": {"rate2": 1e-12, "diffusivity2": 6e-15},
        # "EVA_mPNPNP_02": {**eq_c, **{"ratio": 0.5}},
        # "EVA_mPNPNP_03": {**eq_c, **{"ratio": 1}},
        
        # "EVA_mPNPNP_04": {**eq_c, **{"ratio": 0.01}},
        # "EVA_mPNPNP_05": {**eq_c, **{"ratio": 0.05}},
        # "EVA_mPNPNP_06": {**eq_c, **{"ratio": 0.1}},  

        # "EVA_mPNPNP_07": {**eq_c, **{"m_eq2": 4.48e18}},
        # "EVA_mPNPNP_08": {**eq_c, **{"m_eq2": 2.24e18}},
        # "EVA_mPNPNP_09": {**eq_c, **{"m_eq2": 1.12e18}},
        
        # "EVA_mPNPNP_10": {**eq_c, **{"ratio": 0.1, "rate2": 1e-8}},
        # "EVA_mPNPNP_11": {**eq_c, **{"ratio": 0.1, "rate2": 1e-12}},
        # "EVA_mPNPNP_12": {**eq_c, **{"ratio": 0.1, "rate2": 1e-16}},
        
        # "EVA_mPNPNP_13": {**eq_c, **{"ratio": 0.1, "diffusivity2": 2e-15}},
        # "EVA_mPNPNP_14": {**eq_c, **{"ratio": 0.1, "diffusivity2": 4e-15}},
        # "EVA_mPNPNP_15": {**eq_c, **{"ratio": 0.1, "diffusivity2": 8e-15}},

        # "EVA_mPNPNP_16": {**eq_c, **{"ratio": 0.1, "diffusivity2": 1e-14}},
        # "EVA_mPNPNP_17": {**eq_c, **{"ratio": 0.1, "diffusivity2": 2e-14}},
        "EVA_mPNPNP_18": {**eq_c, **{"ratio": 0.1, "diffusivity2": 3e-14}},         
        
        
        ### R1
        
        # "EVA_mPNPNP_01": {},
        # "EVA_mPNPNP_02": {"ratio": 0.5},
        # "EVA_mPNPNP_03": {"ratio": 1},
        
        # "EVA_mPNPNP_04": {"ratio": 0.01},
        # "EVA_mPNPNP_05": {"ratio": 0.05},
        # "EVA_mPNPNP_06": {"ratio": 0.1},  

        # "EVA_mPNPNP_07": {**eq_c, **{"m_eq2": 4.48e18}},
        # "EVA_mPNPNP_08": {**eq_c, **{"m_eq2": 2.24e18}},
        # "EVA_mPNPNP_09": {**eq_c, **{"m_eq2": 1.12e18}},
        
        # "EVA_mPNPNP_10": {**eq_c, **{"rate2": 1e-8}},
        # "EVA_mPNPNP_11": {**eq_c, **{"rate2": 1e-12}},
        # "EVA_mPNPNP_12": {**eq_c, **{"rate2": 1e-16}},
        
        # "EVA_mPNPNP_13": {**eq_c, **{"rate2": 1e-12, "diffusivity2": 1.2e-15}},
        # "EVA_mPNPNP_14": {**eq_c, **{"rate2": 1e-12, "diffusivity2": 2e-15}},
        # "EVA_mPNPNP_15": {**eq_c, **{"rate2": 1e-12, "diffusivity2": 4e-15}},        
        
        # "EVA_mPNPNP_16": {**eq_c, **{"rate2": 1e-12, "m_eq2": 4.48e18, "diffusivity2": 1.2e-15}},
        # "EVA_mPNPNP_17": {**eq_c, **{"rate2": 1e-12, "m_eq2": 4.48e18, "diffusivity2": 2e-15}},
        # "EVA_mPNPNP_18": {**eq_c, **{"rate2": 1e-12, "m_eq2": 4.48e18, "diffusivity2": 4e-15}},
        
        # "EVA_mPNPNP_19": {**eq_c, **{"rate2": 1e-12, "m_eq2": 4.48e18, "diffusivity2": 8e-15}},
        # "EVA_mPNPNP_20": {**eq_c, **{"rate2": 1e-12, "m_eq2": 4.48e18, "diffusivity2": 1e-14}},
        # "EVA_mPNPNP_21": {**eq_c, **{"rate2": 1e-12, "m_eq2": 4.48e18, "diffusivity2": 2e-14}}, 
        
        }


rpath = p_find("Dropbox (ASU)","Work Docs","Data", "Raw", "Simulations", "PNP", "mPNP_NP_r3", base="home")
pathlib_mk(rpath)
gnote = "mPNP" #"compare calc mPNP and PNP"

for kw, var in runs.items():
    note = gnote + ", ".join([f"{k}={v}" for k, v in var.items()]) 
    file_tag = kw #alternate 4 & 5
    
    h5FileName = str(rpath / Path(f"{file_tag}.h5"))
    myLogger = getLogger(str(rpath), file_tag, name='SL')

    rate = var.pop("rate",  5e-13) #0.5e-12
    rate2 = var.pop("rate2",  rate) #0.5e-12
    
    rm1_val = 4.48e19
    rm2_val = 1e-20
    
    if "ratio" in var.keys():
        ratio = var.pop("ratio")
        var["csurf"] = max(1e-20, c_init * (1 - ratio))
        var["c2surf"] = max(1e-20, c_init - var["csurf"])
        
        var["csurf_vol"] = max(1e-20, cv_init * (1 - ratio))
        var["c2surf_vol"] = max(1e-20, cv_init - var["csurf_vol"])
        
        rm1_val = max(1e-20, var.get("m_eq",  4.48e19) * (1 - ratio))
        rm2_val = max(1e-20, var.get("m_eq",  4.48e19) - rm1_val)
        
    if "mratio" in var.keys():
        rm1_val = max(1e-20, var.get("m_eq",  4.48e19) * (1 - var.pop("mratio")))
        rm2_val = max(1e-20, var.get("m_eq",  4.48e19) - rm1_val)
        
    m1_val = var.pop("m_eq",  rm1_val)
    m2_val = var.pop("m_eq2",  rm2_val)
    
    m_eq = m1_val/var.get("csurf_vol", other["csurf_vol"])
    m_eq2 = m2_val/var.get("c2surf_vol", other["c2surf_vol"])


    inputs = dict(
        note = note,
        in_flux = ("interf", [rate, m_eq]), #("box", 0) or  ("interf",[1,1]) ("surf", rate),
        # out_flux = ("closed", 0), # ("surf", rate), or ("box", 0) or ("interf",[1,1])
        
        in_flux2 = ("interf", [rate2, m_eq2]),
        # out_flux2 = ("closed", 0),
        
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
    
    t_sim, x1i, c1i, p1i, c_max = pnpfs.double_ion(**inputs)
    print("")

    if 0:
        df = pd.DataFrame(np.array([x1i, c1i, p1i]).T, columns = ["depth","conc","voltage"])
        df.plot(x="depth", y="conc", logy=True, grid=True)
        df.plot(x="depth", y="voltage", grid=True)

