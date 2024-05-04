# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:16:32 2022

@author: j2cle
"""
import numpy as np
import dolfin as dlf
from dolfin import (
    near,
    split,
    inner,
    grad,
    div,
    sqrt,
    dot,
    gt,
    exp,
)
import logging
# import pnptransport.utils as utils
import sympy as sp
import h5py
import os
from typing import Union
import scipy.constants as constants
from scipy import integrate
from scipy.stats import linregress
from datetime import datetime as date
import time

# from utilities import get_filenames
from research_tools.functions import f_find

q_red = 1.6021766208  # x 1E-19 C
e0_red = 8.854187817620389  # x 1E-12 C^2 / J m
CM3TOUM3 = 1e-12

def format_time_str(time_s: float):
    """
    Returns a formatted time string

    Parameters
    ----------
    time_s : float
        The time in seconds


    Returns
    -------
    timeStr : str
        A string representing the time
    """
    time_s = abs(time_s)
    min2s = 60
    hr2s = min2s * 60
    day2s = hr2s * 24
    mon2s = day2s * 30.42
    yr2s = day2s * 365

    years = np.floor(time_s / yr2s)
    time_s = time_s - years * yr2s
    months = np.floor(time_s / mon2s)
    time_s = time_s - months * mon2s
    days = np.floor(time_s / day2s)
    time_s = time_s - days * day2s
    hrs = np.floor(time_s / hr2s)
    time_s = time_s - hrs * hr2s
    mins = np.floor(time_s / min2s)
    time_s = time_s - mins * min2s

    #    time_str = "%01d Y %02d M %02d d %02d:%02d:%02d" % (years, \
    #        months,days,hrs,mins,time_s)
    if years >= 1:
        time_str = "%01dY %02dM %02dd %02d:%02d:%02d" % (years,
                                                         months, days, hrs, mins, time_s)
    elif months >= 1:
        time_str = "%02dM %02dd %02d:%02d:%02d" % (months, days, hrs, mins, time_s)
    elif days >= 1:
        time_str = "%02dd %02d:%02d:%02d" % (days, hrs, mins, time_s)
    elif hrs >= 1:
        time_str = "%02d:%02d:%02d" % (hrs, mins, time_s)
    elif mins >= 1:
        time_str = "%02d:%02d" % (mins, time_s)
    else:
        time_str = "%02d" % time_s
    return time_str

def double_ion(
    tempC: float,
    voltage: float,
    thick: float,
    time_s: Union[float, int],
    diffusivity: float,
    er: float = 7,
    material: str = "layer",
    csurf: float = 1e11,
    cbulk: float = 1e-20,
    fcall: int = 1,
    **kwargs,
):
    """
    This function simulates the flatband voltage as a function of time for a
    MIS device where Na is migrating into the cell. It also returns a matrix
    with the concentration profiles as a function of time.

    The system solves Poisson-Nernst-Planck equation for a single species.

    *Example*

    .. code-block:: python

        import pnptransport.finitesource as pnpfs
        import logging

        D1 = 1E-16
        thickness_1 = 75E-7
        temp_c = 60.
        s, k = 1E10, 1E-6
        voltage = 0.75
        time_s = 86400.
        h5FileName = 'simulation_output.h5'
        # Chose a small time step to reduce truncation error in the TRBDF2
        t_steps = 3600

        # Create a logger
        logFile = 'simulation_output.log'
        my_logger = logging.getLogger('simlog')
        my_logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(logFile)
        fh.setLevel(logging.DEBUG)
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # add the handlers to the logger
        my_logger.addHandler(fh)
        my_logger.addHandler(ch)

        tsim, x1, c1_trial_ti1, x2, c2, cmax = pnpfs.double_ion_constant_source_flux(
            diffusivity=D1, thick=thickness_1,
            tempC=temp_c, voltage=voltage, time_s=csurf,
            rate=k, tpoints=t_steps, h5_storage=h5FileName, er=7.0
        )




    Parameters
    ----------
    diffusivity: float
        The diffusion coefficient of Na in the dielectric (cm\ :sup:`2`\/s)
    thick: float
        The thickness of the simulated dielectric layer (um)
    tempC: Union[float, int]
        The temperature in °C
    voltage: Union[float, int]
        The voltage applied to the dielectric (V)
    time_s: Union[float, int]
        The simulation time in seconds
    csurf: float
        The initial surface concentration in atoms/cm\ :sup:`2`\.
        The flux at the source will be determined as J = rate * suface_concentration.
        Default value 1E11 atoms/cm\ :sup:`2`\.
    rate: float
        The rate of transfer at the source in 1/s
        Defulat 1E-5
    **kwargs:
        cbulk: double
            The base concentration cm\ :sup:`-3`.
        xpoints: int
            The number of cells in the sinx layer
        valence: integer
            The valence of the ion
            default: 1
        er: double
            The relative permittivity of the dielectric
        xpoints: int
            The number of x points to simulate
        fcall: int
            The number of times the function has been called to solve the same
            problem
        tpoints: int
            The number of time steps to simulate
        max_calls: int
            The maximum number of times the function can be recursively call if the convergence fails.
        max_iter: int
            The maximum number of iterations for the solver
        relax_param: float
            The relaxation w for the Newton algorithm
        h5fn: str
            The path to the h5 file to store the simulation results
        debug: bool
            True if debugging the function

    Returns
    -------
    Vfb: np.ndarray
        An array containing the flat band voltage shift as a function of time
        in (V)
    tsim: np.ndarray
        The time for each flatband voltage point in seconds.
    x1: np.ndarray
        The depth of the concentration profile in SiNx in um.
    c1_trial_ti1: np.ndarray
        The final concentration profile as a function of depth in SiNx in cm\ :sup:`-3`\.
    potential: np.ndarray
        The final potential profile as a function of depth in SiNx in V.
    cmax: float
        The maximum concentration in silicon nitride in cm\ :sup:`-3`\.
    """
    #%% parameter imports
    # q_red = 1.6021766208  # x 1E-19 C
    # e0_red = 8.854187817620389  # x 1E-12 C^2 / J m
    voltage_pin = kwargs.get("voltage_pin", True)
    thick_mesh = kwargs.get("thick_mesh", thick)
    thick_mesh_ref = kwargs.get("thick_mesh_ref", thick_mesh * 0.1)
    thick_na = kwargs.get("thick_na", 0.0)
    
    csurf_vol = kwargs.get("csurf_vol", csurf)
    in_flux = kwargs.get("in_flux", ("surface", 1e-4))
    out_flux = kwargs.get("out_flux", ("closed", 0))
    
    diffusivity2 = kwargs.get("diffusivity2", diffusivity)
    c2surf = kwargs.get("c2surf", csurf)
    c2bulk = kwargs.get("c2bulk", cbulk)
    c2surf_vol = kwargs.get("c2surf_vol", csurf_vol)
    in_flux2 = kwargs.get("in_flux2", in_flux)
    out_flux2 = kwargs.get("out_flux2", out_flux)
    
    dx_max = kwargs.get("dx_max", 1e-7)
    dx_min = kwargs.get("dx_min", dx_max * 0.01)
    dt_max = kwargs.get("dt_max", int(time_s / 500))
    dt_base = kwargs.get("dt_base", 3)
    max_calls = kwargs.get("max_calls", 5)
    max_iter = kwargs.get("max_iter", 1000)

    valence = kwargs.get("valence", 1.0)
    h5fn = kwargs.get("h5_storage", None)
    debug = kwargs.get("debug", False)
    relax_param = kwargs.get("relax_param", 1.0)
    note = kwargs.get("note", "None")
    func = kwargs.get("func", None)
    screen = kwargs.get("screen", None)
    
    tm_start = time.time()
    
    # %% Base value calculations
    
    # Estimate the diffusion coefficients for the given temperature
    tempK = tempC + 273.15

    epsilon = constants.epsilon_0 * er / 100  # C/Vcm
    zq = valence * constants.elementary_charge  # C
    
    qee = zq / epsilon # Vcm
    
    kbT = constants.Boltzmann * tempK  # CV
            
    mob = zq * diffusivity / kbT
    
    if isinstance(screen, str):
        if "calc" in screen.lower() and "inter" in in_flux[0].lower():
            screen = np.sqrt(epsilon*kbT/(zq**2*in_flux[1][1]*csurf_vol))*1e7  # cm
        elif "none" in screen.lower():
            screen = None

    # %% sympy stuff
    # Generate the initial profile
    f = sp.symbols("f", cls=sp.Function)
    x = sp.Symbol("x[0]", real=True)

    ode_vbias = sp.Eq(f(x).diff(x, x), 0)
    f_vbias = sp.dsolve(ode_vbias, f(x), simplify=False, ics={f(0): voltage, f(thick): 0})

    if thick_na == 0:
        thick_na  = dx_max/2**int(np.log(dx_max / dx_min) / np.log(2))*2
        conc = sp.Piecewise(
            (0, x < 0),
            (csurf, x <= thick_na),  # CM3TOUM3
            (cbulk, x <= thick),  # CM3TOUM3
            (0, x > thick),
            evaluate=False,
        )
        
        conc2 = sp.Piecewise(
            (0, x < 0),
            (c2surf, x <= thick_na),  # CM3TOUM3
            (c2bulk, x <= thick),  # CM3TOUM3
            (0, x > thick),
            evaluate=False,
        )

        ion_dens = sp.Piecewise(
            (0, x < 0),
            (qee * csurf, x <= thick_na),  # CM3TOUM3
            (qee * cbulk, x <= thick),  # CM3TOUM3
            (0, x > thick),
            evaluate=False,
        )

    else:
        conc = sp.Piecewise(
            (0, x < 0),
            (csurf_vol, x <= thick_na),  # CM3TOUM3
            (cbulk, x <= thick),  # CM3TOUM3
            (0, x > thick),
            evaluate=False,
        )
        
        conc2 = sp.Piecewise(
            (0, x < 0),
            (c2surf_vol, x <= thick_na),  # CM3TOUM3
            (c2bulk, x <= thick),  # CM3TOUM3
            (0, x > thick),
            evaluate=False,
        )

        ion_dens = sp.Piecewise(
            (0, x < 0),
            (qee * csurf_vol, x <= thick_na),  # CM3TOUM3
            (qee * cbulk, x <= thick),  # CM3TOUM3
            (0, x > thick),
            evaluate=False,
        )
    if ion_dens.simplify() == 0:
        f_vs = f_vbias
    else:
        ode_v = sp.Eq(f(x).diff(x, x), -1 * ion_dens)

        eion_s = sp.integrate(x / thick * conc, (x, 0, thick)) * -1 * qee

        f_vs = sp.dsolve(
            ode_v,
            f(x),
            simplify=False,
            ics={f(x).diff(x).subs(x, thick): float((f_vbias.rhs.diff(x) + eion_s)), f(thick): 0},
        )        
 
    # %% Logger
    fcallLogger = logging.getLogger("simlog")

    # Chose the backend type
    if dlf.has_linear_algebra_backend("PETSc"):
        dlf.parameters["linear_algebra_backend"] = "PETSc"
    #        print('PETSc linear algebra backend found.')
    elif dlf.has_linear_algebra_backend("Eigen"):
        dlf.parameters["linear_algebra_backend"] = "Eigen"
    else:
        fcallLogger.warning("DOLFIN has not been configured with PETSc or Eigen.")
        exit()
    
    dlf.set_log_level(50)
    logging.getLogger("FFC").setLevel(logging.WARNING)
    
    
    if debug:
        
        fcallLogger.info("********* Global parameters *********")
        fcallLogger.info("Start: {}".format(date.now().strftime("%D_%H:%M:%S")))
        fcallLogger.info("-------------------------------------")
        fcallLogger.info("Time: {0}".format(format_time_str(time_s)))
        fcallLogger.info("Temperature: {0:.1f} °C".format(tempC))
        if "inter" in in_flux[0].lower():
            fcallLogger.info("Source concentration: {0:.4E} (Na atoms/cm^3)".format(csurf_vol))
            fcallLogger.info("Equalibrium concentration: {0:.4E} (Na atoms/cm^3)".format(in_flux[1][1]*csurf_vol))
        else:
            fcallLogger.info("Source concentration: {0:.4E} (Na atoms/cm^2)".format(csurf))
        if "inter" in in_flux2[0].lower():
            fcallLogger.info("Source concentration: {0:.4E} (Na atoms/cm^3)".format(c2surf_vol))
            fcallLogger.info("Equalibrium concentration: {0:.4E} (Na atoms/cm^3)".format(in_flux2[1][1]*c2surf_vol))
        else:
            fcallLogger.info("Source concentration: {0:.4E} (Na atoms/cm^2)".format(c2surf))
        fcallLogger.info("*************** {} ******************".format(material))
        fcallLogger.info("Material Thickness: {0:.3G} um".format(thick * 1e4))
        fcallLogger.info("Mesh Thickness: {0:.3G} um".format(thick_mesh * 1e4))
        fcallLogger.info("er: {0:.2f}".format(er))
        fcallLogger.info("Voltage: {0:.1f} V".format(voltage))
        fcallLogger.info("Electric Field: {0:.4E} MV/cm".format(voltage / thick * 1e-6))
        fcallLogger.info("D1: {0:.4E} cm^2/s".format(diffusivity))
        fcallLogger.info("D2: {0:.4E} cm^2/s".format(diffusivity2))
        fcallLogger.info("Ionic mobility: {0:.4E} um^2/ V*s".format(mob * 1e8))

    # %% make dt's
    dt = dt_max
    num_t = int(np.log10(dt) * 2 * dt_base)
    dtt = [dt_max / (dt_base * k) for k in range(1, num_t)]
    t1 = []
    for k in dtt:
        dt -= k
        t1.append(dt)
    t1 = np.array(t1)[np.array(t1) > 0][::-1]
    t2 = np.array([k * dt_max for k in range(1, int(time_s / dt_max) + 1)], dtype=np.float64)
    t_sim = np.concatenate([[0], t1, t2])
    dtt = np.concatenate([np.diff(t_sim), [dt_max]])

    if t_sim[-1] != time_s:
        t_sim = np.append(t_sim, time_s)
        dtt = np.append(dtt, np.diff(t_sim)[-1])

    del dt, num_t, t1, t2

    # %% Dolphin FEM Meshing

    class Top(dlf.SubDomain):
        def inside(self, x_, on_boundary):
            return near(x_[0], 0.0, 1e-12) and on_boundary

    class Bottom(dlf.SubDomain):
        def inside(self, x_, on_boundary):
            return near(x_[0], thick_mesh, 1e-12) and on_boundary

    def get_solution_array1(mesh, sol):
        c_, phi = sol.split()
        xu = mesh.coordinates()
        cu = c_.compute_vertex_values(mesh)  # * 1e12
        pu = phi.compute_vertex_values(mesh)
        xyz = np.array(
            [(xu[j], cu[j], pu[j]) for j in range(len(xu))],
            dtype=[("x", "d"), ("c", "d"), ("phi", "d")],
        )
        xyz.sort(order="x")
        return xyz["x"], xyz["c"], xyz["phi"]
    
    def get_solution_array2(mesh, sol):
        xu = mesh.coordinates()
        cu = sol.compute_vertex_values(mesh)  # * 1e12
        xyz = np.array(
            [(xu[j], cu[j]) for j in range(len(xu))],
            dtype=[("x", "d"), ("c", "d")],
        )
        xyz.sort(order="x")
        return xyz["x"], xyz["c"]
    
    top = Top()
    bottom = Bottom()

    # Create mesh and define function space
    xpoints = int(thick_mesh / dx_max)
    mesh1 = dlf.IntervalMesh(xpoints, 0.0, thick_mesh)
    mesh2 = dlf.IntervalMesh(xpoints, 0.0, thick_mesh)
    
    nor = int(np.log(dx_max / dx_min) / np.log(2))
    if debug:
        nor_ranges = []
    for i in range(nor):
        cell_markers = dlf.MeshFunction("bool", mesh1, mesh1.topology().dim(), False)
        for cell in dlf.cells(mesh1):
            p = cell.midpoint()
            if p[0] >= thick_mesh - thick_mesh_ref or p[0] <= thick_mesh_ref:
                cell_markers[cell] = True
        mesh1 = dlf.refine(mesh1, cell_markers)
        if debug:
            nor_ranges.insert(
                0, "dx={0:.1E}um to x={1:.1E}um".format(mesh1.hmin() * 1e4, thick_mesh_ref * 1e4)
            )
        
        cell_markers = dlf.MeshFunction("bool", mesh2, mesh2.topology().dim(), False)
        for cell in dlf.cells(mesh2):
            p = cell.midpoint()
            if p[0] >= thick_mesh - thick_mesh_ref or p[0] <= thick_mesh_ref:
                cell_markers[cell] = True
        mesh2 = dlf.refine(mesh2, cell_markers)

        thick_mesh_ref = thick_mesh_ref / 1.5
        

    
    # %% Dolphin FEM Meshing
    # Initialize mesh function for boundary domains
    mesh_func = dlf.MeshFunction("size_t", mesh1, mesh1.topology().dim() - 1)
    mesh_func.set_all(0)

    top.mark(mesh_func, 1)
    bottom.mark(mesh_func, 2)

    # Define the measures
    ds1 = dlf.Measure("ds", domain=mesh1, subdomain_data=mesh_func)
    dx1 = dlf.Measure("dx", domain=mesh1, subdomain_data=mesh_func)
    
    # %% Dolphin FEM Meshing
    # Initialize mesh function for boundary domains
    mesh_func2 = dlf.MeshFunction("size_t", mesh2, mesh2.topology().dim() - 1)
    mesh_func2.set_all(0)

    top.mark(mesh_func2, 1)
    bottom.mark(mesh_func2, 2)

    # Define the measures
    ds2 = dlf.Measure("ds", domain=mesh2, subdomain_data=mesh_func2)
    dx2 = dlf.Measure("dx", domain=mesh2, subdomain_data=mesh_func2)

    # %% Dolphin FEM meshing
    f1_trial_t0 = dlf.Expression(
        (sp.printing.ccode(conc.evalf()), sp.printing.ccode(f_vs.rhs)), degree=1
    )

    # Defining the mixed function space
    CG1 = dlf.FiniteElement("CG", mesh1.ufl_cell(), 1)
    W1 = dlf.FunctionSpace(mesh1,  dlf.MixedElement([CG1, CG1]))

    # Defining the "Trial" functions
    f1_trial_ti1 = dlf.interpolate(f1_trial_t0, W1)  # For time i+1
    c1_trial_ti1, phi1_trial_ti1 = split(f1_trial_ti1)
    f1_trial_tig = dlf.interpolate(f1_trial_t0, W1)  # For time i+1/2
    c1_trial_tig, phi1_trial_tig = split(f1_trial_tig)
    f1_trial_ti0 = dlf.interpolate(f1_trial_t0, W1)  # For time i
    c1_trial_ti0, phi1_trial_ti0 = split(f1_trial_ti0)

    # Define the test functions
    (c1_test, phi1_test) = split(dlf.TestFunction(W1))

    du1 = dlf.TrialFunction(W1)

    f1_trial_ti1.set_allow_extrapolation(True)
    f1_trial_tig.set_allow_extrapolation(True)
    f1_trial_ti0.set_allow_extrapolation(True)
    
    # %% Dolphin FEM meshing
    # f2_trial_t0 = dlf.Expression('cb', cb=c2bulk, degree=0)
    f2_trial_t0 = dlf.Expression(sp.printing.ccode(conc2.evalf()), degree=0)

    W2 = dlf.FunctionSpace(mesh2, 'CG', 1)

    # Defining the "Trial" functions
    c2_trial_ti1 = dlf.interpolate(f2_trial_t0, W2)  # For time i+1
    c2_trial_tig = dlf.interpolate(f2_trial_t0, W2)  # For time i+1/2
    c2_trial_ti0 = dlf.interpolate(f2_trial_t0, W2)  # For time i


    # Define the test functions
    c2_test = dlf.TestFunction(W2)
    
    du2 = dlf.TrialFunction(W2)

    c2_trial_ti1.set_allow_extrapolation(True)
    c2_trial_tig.set_allow_extrapolation(True)
    c2_trial_ti0.set_allow_extrapolation(True)
    # %% Dolphin FEM bc
    def mid_bias(x_arr, y_arr, l_pnts, x2):
        # array of len x_arr with true for linear region
        ind = np.argmax(np.diff(y_arr) / np.diff(y_arr)[int(len(y_arr) * -0.01):].mean())
        (
            m,
            b,
            _,
            _,
            _,
        ) = linregress([x_arr[ind], l_pnts[0]], [y_arr[ind], l_pnts[1]])
        return m * x2 + b

    def update_bcs(bias_0, bias_L=0.0, pin=False):
        bcs_ = [dlf.DirichletBC(W1.sub(1), bias_L, mesh_func, 2)]
        if pin:
            bcs_.insert(0, dlf.DirichletBC(W1.sub(1), bias_0, mesh_func, 1))

        if "const" in in_flux[0].lower():
            bcs_.insert(0, dlf.DirichletBC(W1.sub(0), csurf_vol, mesh_func, 1))
        return bcs_

    volt_mesh0 = (1 - thick_mesh / thick) * voltage
    bcs = update_bcs(voltage, volt_mesh0, voltage_pin)
    
    bcs2 = None
    
    thick_flux = diffusivity / mesh1.hmin() / 10

    # %% Dolphin FEM functions
    def get_variational_form1(c_ftrial, phi_ftrial, gp1_, gp2_, time_i):
        """
        Generate voltage bc
        
        non-imported terms:
            csurf, csurf_vol, in_flux
            screen, zq, kbT, mob
            c1_test, phi1_test
            dx1, ds1
                        
        """
        c_grad_01 = -mob * c_ftrial * gp1_
        if "surf" in in_flux[0].lower():
            c_grad_01 += in_flux[1] * csurf  # * 1e-8
        elif "inter" in in_flux[0].lower():
            c_grad_01 += in_flux[1][0] * (csurf_vol - c_ftrial / in_flux[1][1])  # * 1e4
        elif "const" in in_flux[0].lower():  # or "block" in in_flux[0].lower():
            c_grad_01 = 0.0

        c_grad_12 = -mob * c_ftrial * gp2_
        if "surf" in out_flux[0].lower():
            c_grad_12 -= out_flux[1] * csurf  # * 1e-8
        elif "inter" in out_flux[0].lower():
            c_grad_12 -= out_flux[1][0] * (csurf_vol - c_ftrial / out_flux[1][1])  # * 1e4
        elif "const" in out_flux[0].lower():  # or "block" in out_flux[0].lower():
            c_grad_12 = 0.0
        elif thick != thick_mesh:
            c_grad_12 -= thick_flux * c_ftrial
        
        if isinstance(screen, str):
            scr_exp = exp(-1 * zq * phi_ftrial / kbT)
        elif screen is None or screen == 0:
            scr_exp = 1
        else:
            scr_exp = dlf.Expression("exp(-1*kd*x[0])", kd=abs(1/(screen*1e-7)), degree=1)
        
        # Base NP
        a = -diffusivity * inner(grad(c_ftrial), grad(c1_test)) * dx1  # Anp term 1
        a += c_grad_01 * c1_test * ds1(1)  # Anp term2 in; i.e. bc in
        a += c_grad_12 * c1_test * ds1(2)  # Anp term2 out; i.e. bc out
        # W/ bias
        a -= mob * c_ftrial * inner(grad(phi_ftrial), grad(c1_test)) * dx1  # Anp term 3
        a += mob * gp1_ * c_ftrial * c1_test * ds1(1)  # Anp term 4 in
        a += mob * gp2_ * c_ftrial * c1_test * ds1(2)  # Anp term 4 out
        
        # W/ Poisson
        a -= inner(grad(phi_ftrial), grad(phi1_test)) * dx1  # Ap term 1
        a += gp1_ * phi1_test * ds1(1)  # Ap term 2 in
        a += gp2_ * phi1_test * ds1(2)  # Ap term 2 out
        a += qee * c_ftrial * phi1_test * scr_exp * dx1  # Ap term 3
        return a
    
    def get_variational_form2(c_ftrial):

        if "surf" in in_flux2[0].lower():
            c_grad_01 = in_flux2[1] * c2surf  # * 1e-8
        elif "inter" in in_flux2[0].lower():
            c_grad_01 = in_flux2[1][0] * (c2surf_vol - c_ftrial / in_flux2[1][1])  # * 1e4
        else: 
            c_grad_01 = 0.0


        if "surf" in out_flux2[0].lower():
            c_grad_12 = -out_flux2[1] * c2surf  # * 1e-8
        elif "inter" in out_flux2[0].lower():
            c_grad_12 = -out_flux2[1][0] * (c2surf_vol - c_ftrial / out_flux2[1][1])  # * 1e4
        elif thick != thick_mesh:
            c_grad_12 = -thick_flux * c_ftrial
        else:
            c_grad_12 = 0.0
            
        # Base NP
        a = -diffusivity2 * inner(grad(c_ftrial), grad(c2_test)) * dx2  # Anp term 1
        a += c_grad_01 * c2_test * ds2(1)  # Anp term2 in; i.e. bc in
        a += c_grad_12 * c2_test * ds2(2)  # Anp term2 out; i.e. bc out
        return a
    
    def getTRBDF2ta(c_ftrial, phi_ftrial):
        if isinstance(screen, str):
            scr_exp = exp(-1 * zq * phi_ftrial / kbT)
        elif screen is None or screen == 0:
            scr_exp = 1
        else:
            scr_exp = dlf.Expression("exp(-1*kd*x[0])", kd=abs(1/(screen*1e-7)), degree=1)
            
        r = (
            diffusivity * div(grad(c_ftrial))  # Lnp term 1
            + mob * c_ftrial * div(grad(phi_ftrial))  # Lnp term 2
            + mob * inner(grad(phi_ftrial), grad(c_ftrial))  # Lnp term 3
            + div(grad(phi_ftrial))  # Lp term 1
            + qee * c_ftrial * scr_exp  # Lp term 2
        )
        return r

    def update_potential_bc(uui, bias: float = voltage):
        """
        Generate voltage bc
        
        non-imported terms:
            mesh1
            screen, zq, kbT, valence, thick, bias, er
            
        """
        uxi, uci, upi = get_solution_array1(mesh1, uui)
        
        if isinstance(screen, str):
            scr_exp = np.exp(-1 * zq * upi / kbT)
        elif screen is None or screen == 0:
            scr_exp = 1
        else:
            scr_exp = np.exp(-1 * abs(1/(screen*1e-7)) * uxi)

        # exp_val = -1 * constants.e * valence / (constants.Boltzmann * tempK)
        
        # The integrated concentration in the oxide (cm-2) <- Check: (1/cm^3) x (um) x (1E-4 cm/1um)
        Ctot_ = integrate.simps(uci * scr_exp, uxi)  # * 1e-4
        # Ctot_ = integrate.simps(uci * np.exp(exp_val * upi), uxi)
        # Ctot_ = np.nan_to_num(integrate.simpson(uci * (np.exp(exp_val * upi)-np.exp(-exp_val * upi)), uxi))
        # The integral in Poisson's equation solution (Nicollian & Brews p.426)
        # units: 1/cm <------------- Check: (1/cm^3) x (um^2) x (1E-4 cm/1um)^2
        Cint_ = integrate.simps(uxi * uci * scr_exp, uxi)  # * 1e-8
        # Cint_ = integrate.simps(uxi * uci * np.exp(exp_val * upi), uxi)
        # Cint_ = np.nan_to_num(integrate.simpson(uxi * uci * (np.exp(exp_val * upi)-np.exp(-exp_val * upi)), uxi))
        # The centroid of the charge distribution
        xbar_ = uxi.mean()
        if Ctot_ != 0:
            xbar_ = Cint_ / Ctot_  # * 1e4

        # The surface charge density at silicon C/cm2
        scd_si = -1 * constants.e * valence * (xbar_ / thick) * Ctot_
        # scd_si2 = -1 * constants.e * valence * integrate.simps( uxi / thick * uci * np.exp(screen * uxi), uxi)
        # The surface charge density at the gate C/cm2
        scd_g = -1 * constants.e * valence * (1.0 - xbar_ / thick) * Ctot_
        # scd_g2 = constants.e * valence * integrate.simps((uxi - thick) / thick * uci * np.exp(screen * uxi), uxi)
        uei = np.diff(upi)/np.diff(uxi)
        # The applied electric field in V/cm
        field_app = bias / thick
        # The electric field at the gate interface V/um
        # (C / cm^2) * (J * m / C^2 ) x ( 1E2 cm / 1 m) x ( 1E cm / 1E4 um)
        gp1_ = field_app + scd_g / (constants.epsilon_0 * er) * 100  # x
        # gp1_2 = field_app + scd_g2 / (constants.epsilon_0 * er) * 100  # x
        # The electric field at the Si interface V/um
        gp2_ = -(field_app - scd_si / (constants.epsilon_0 * er) * 100)  # x
        # gp2_2 = -(field_app - scd_si2 / (constants.epsilon_0 * er) * 100)  # x

        return gp1_, gp2_, Cint_, Ctot_, xbar_

    # %% Dolphin FEM stuff

    GAMMA = 2.0 - np.sqrt(2.0)  # 0.59
    TRF = dlf.Constant(0.5 * GAMMA)
    BDF2_T1 = dlf.Constant(1.0 / (GAMMA * (2.0 - GAMMA)))
    BDF2_T2 = dlf.Constant((1.0 - GAMMA) * (1.0 - GAMMA) / (GAMMA * (2.0 - GAMMA)))
    BDF2_T3 = dlf.Constant((1.0 - GAMMA) / (2.0 - GAMMA))

    ffc_options = {"optimize": True, "cpp_optimize": True, "quadrature_degree": 5}

    newton_solver_parameters = {
        "nonlinear_solver": "newton",
        "newton_solver": {
            "linear_solver": "lu",
            # "preconditioner": 'ilu',  # 'hypre_euclid',
            "convergence_criterion": "incremental",
            "absolute_tolerance": 1e-5,
            "relative_tolerance": 1e-4,
            "maximum_iterations": max_iter,
            "relaxation_parameter": relax_param,
        },
    }
    
    def SUPG_terms(trial_t0, trial_t1):
        b_ = mob * dlf.Dx(trial_t0, 0)
        nb_ = sqrt(dot(b_, b_) + dlf.DOLFIN_EPS)
        Pek = nb_ * dlf.CellDiameter(mesh1) / (2.0 * diffusivity)

        tau_ = dlf.conditional(
            gt(Pek, dlf.DOLFIN_EPS),
            (dlf.CellDiameter(mesh1) / (2.0 * nb_))
            * (((exp(2.0 * Pek) + 1.0) / (exp(2.0 * Pek) - 1.0)) - 1.0 / Pek),
            0.0,
        )

        Lss_ = (
            mob * inner(grad(trial_t1), grad(c1_test))
            + (mob / 2) * div(grad(trial_t1)) * c1_test
        )
        
        return tau_, Lss_


    def solve_1G(gp1_, gp2_, dt_, time_i):
        """
        Generate dlf solvers
        
        non-imported terms:
            f1_trial_ti0, f1_trial_tig, f1_trial_ti1
            c1_test
            dx1
            TRF, BDF2_T1, BDF2_T2, BDF2_T3
            mob, diffusivity
            f1_trial_tig, f1_trial_ti1, du1
            bcs, ffc_options, newton_solver_parameters
            
        """
        a10 = get_variational_form1(c1_trial_ti0, phi1_trial_ti0, gp1_, gp2_, time_i)
        a1G = get_variational_form1(c1_trial_tig, phi1_trial_tig, gp1_, gp2_, time_i)

        F1G = (1.0 / dt_) * (c1_trial_tig - c1_trial_ti0) * c1_test * dx1 - TRF * (a1G + a10)

        tau, Lss = SUPG_terms(phi1_trial_ti0, phi1_trial_tig)
        
        # SUPG Stabilization term
        ta = getTRBDF2ta(c1_trial_tig, phi1_trial_tig)
        tb = getTRBDF2ta(c1_trial_ti0, phi1_trial_ti0)

        ra = inner(((1 / dt_) * (c1_trial_tig - c1_trial_ti0) - TRF * (ta + tb)), tau * Lss) * dx1

        F1G += ra

        J1G = dlf.derivative(F1G, f1_trial_tig, du1)

        problem1G = dlf.NonlinearVariationalProblem(
            F1G, f1_trial_tig, bcs, J1G, form_compiler_parameters=ffc_options
        )

        solver1G_ = dlf.NonlinearVariationalSolver(problem1G)
        solver1G_.parameters.update(newton_solver_parameters)
        solver1G_.solve()
        
        a20 = get_variational_form2(c2_trial_ti0)
        a2G = get_variational_form2(c2_trial_tig)

        F2G = (1.0 / dt_) * (c2_trial_tig - c2_trial_ti0) * c2_test * dx2 - TRF * (a2G + a20)

        J2G = dlf.derivative(F2G, c2_trial_tig, du2)

        problem2G = dlf.NonlinearVariationalProblem(
            F2G, c2_trial_tig, bcs2, J2G, form_compiler_parameters=ffc_options
        )

        solver2G_ = dlf.NonlinearVariationalSolver(problem2G)
        solver2G_.parameters.update(newton_solver_parameters)
        solver2G_.solve()
    
    def solve_1N(gp1_, gp2_, dt_, time_i):
        """
        Generate dlf solvers
        
        non-imported terms:
            f1_trial_ti0, f1_trial_tig, f1_trial_ti1
            c1_test
            dx1
            TRF, BDF2_T1, BDF2_T2, BDF2_T3
            mob, diffusivity
            f1_trial_tig, f1_trial_ti1, du1
            bcs, ffc_options, newton_solver_parameters
            
        """
        a11 = get_variational_form1(c1_trial_ti1, phi1_trial_ti1, gp1_, gp2_, time_i)

        F1N = (1.0 / dt_) * (
            c1_trial_ti1 - BDF2_T1 * c1_trial_tig + BDF2_T2 * c1_trial_ti0
        ) * c1_test * dx1 - BDF2_T3 * a11
        
        tau, Lss = SUPG_terms(phi1_trial_tig, phi1_trial_ti1)
        
        # SUPG Stabilization term
        tc = getTRBDF2ta(c1_trial_ti1, phi1_trial_ti1)
        rb = (
            inner(
                (
                    c1_trial_ti1 / dt_
                    - BDF2_T1 * c1_trial_tig / dt_
                    + BDF2_T2 * c1_trial_ti0 / dt_
                    - BDF2_T3 * tc
                ),
                tau * Lss,
            )
            * dx1
        )

        F1N += rb

        J1N = dlf.derivative(F1N, f1_trial_ti1, du1)  # J1G

        problem1N = dlf.NonlinearVariationalProblem(
            F1N, f1_trial_ti1, bcs, J1N, form_compiler_parameters=ffc_options
        )
        solver1N_ = dlf.NonlinearVariationalSolver(problem1N)
        solver1N_.parameters.update(newton_solver_parameters)
        solver1N_.solve()
        
        a21 = get_variational_form2(c2_trial_ti1)

        F2N = (1.0 / dt_) * (
            c2_trial_ti1 - BDF2_T1 * c2_trial_tig + BDF2_T2 * c2_trial_ti0
        ) * c2_test * dx2 - BDF2_T3 * a21
        

        J2N = dlf.derivative(F2N, c2_trial_ti1, du2)  # J1G

        problem2N = dlf.NonlinearVariationalProblem(
            F2N, c2_trial_ti1, bcs2, J2N, form_compiler_parameters=ffc_options
        )
        solver2N_ = dlf.NonlinearVariationalSolver(problem2N)
        solver2N_.parameters.update(newton_solver_parameters)
        solver2N_.solve()
        
        

    # %% Get initial vals
    x1i, c1i, p1i = get_solution_array1(mesh1, f1_trial_ti0)
    x2i, c2i = get_solution_array2(mesh2, c2_trial_ti0)
    csi = c1i + c2i
    c_max = -np.inf

    # %% Save to hdf
    if h5fn is not None:
        h5fn_tmp = "".join([h5fn[:-3], date.now().strftime("_%H_%M"), h5fn[-3:]])
        if os.path.exists(h5fn_tmp):
            os.remove(h5fn_tmp)

        with h5py.File(h5fn_tmp, "w") as hf:
            # file_tag = os.path.splitext(os.path.basename(h5fn_tmp))[0]
            if debug:
                fcallLogger.info("Created file for storage '{}'".format(h5fn_tmp))

            dst = hf.create_dataset("/time", (len(t_sim),))
            dst[...] = t_sim
            dst.attrs["t_max"] = time_s

        with h5py.File(h5fn_tmp, "a") as hf:
            grp_l1 = hf.create_group(material)

            dsx1 = grp_l1.create_dataset("x", (len(x1i),))
            dsx1[...] = x1i
            grp_l1.attrs["Material"] = material
            grp_l1.attrs["L"] = thick
            grp_l1.attrs["csurf"] = csurf
            grp_l1.attrs["cbulk"] = cbulk
            grp_l1.attrs["D"] = diffusivity
            grp_l1.attrs["D2"] = diffusivity2
            grp_l1.attrs["er"] = er
            grp_l1.attrs["T"] = tempC
            grp_l1.attrs["V"] = voltage
            grp_l1.attrs["E"] = voltage / thick * 1e-6

            for kw, va in kwargs.items():
                if kw in ["h5_storage", "relax_param", "func", "debug", "fcallLogger"]:
                    continue
                if isinstance(va, tuple) and isinstance(va[0], str):
                    grp_l1.attrs[f"{kw}_{va[0]}"] = va[1]
                elif "screen" in kw.lower() and va is None:
                    grp_l1.attrs[kw] = "None"
                elif "screen" in kw.lower() and not isinstance(screen, str):
                    grp_l1.attrs[kw] = screen  
                else:
                    grp_l1.attrs[kw] = va

            grp_l1.create_group("concentration")
            grp_l1.create_group("potential")


    # %% Logger
    if debug:
        fcallLogger.info("********** Mesh 1 **********")
        fcallLogger.info("Elements: {0}".format(len(mesh1.coordinates())))
        fcallLogger.info(
            "MIN DX: {0:.3E} um, MAX DX {1:.3E} um".format(mesh1.hmin() * 1e4, mesh1.hmax() * 1e4)
        )
        fcallLogger.info("DX Ranges: {0}".format(", ".join(nor_ranges)))
        fcallLogger.info("**** Time stepping *****")
        fcallLogger.info(
            "Min dt: {0}, Max dt: {1}.".format(
                format_time_str(np.amin(dtt)), format_time_str(np.amax(dtt))
            )
        )
        fcallLogger.info("Simulation time: {0}.".format(format_time_str(time_s)))
        fcallLogger.info("Number of time steps: {0}".format(len(t_sim)))
        fcallLogger.info("Starting time integration loop...")

    # %% Loop
    """
    This section starts the integration loop
    """    

    for n, t in enumerate(t_sim):
        dti = dtt[n]

        volt_intf = 0.0
        if thick != thick_mesh:
            volt_intf = mid_bias(x1i, p1i, [thick, 0.0], thick_mesh)

        if p1i[0] < voltage:
            bcs = update_bcs(voltage, volt_intf, True)
        else:
            bcs = update_bcs(voltage, volt_intf, voltage_pin)

        gp1, gp2, Cint, Ctot, xbar = update_potential_bc(f1_trial_ti0, bias=voltage)
        
        x1i, c1i, p1i = get_solution_array1(mesh1, f1_trial_ti0)
        x2i, c2i = get_solution_array2(mesh2, c2_trial_ti0)
        csi = c1i + c2i
        
        c_max = max(c_max, np.amax(c1i))
        thick_flux = thick_flux * c1i[-1] / cbulk
        if func is not None:
            func(x1i * 1e4, csi, p1i)

        if h5fn is not None:
            # Store the data in h5py
            with h5py.File(h5fn_tmp, "a") as hf:
                grp_l1_c = hf["{}/concentration".format(material)]
                grp_l1_p = hf["{}/potential".format(material)]
                dsc_str = "ct_{0:d}".format(n)
                dsv_str = "vt_{0:d}".format(n)
                if dsc_str not in grp_l1_c:
                    grp_l1_c.attrs["t_final"] = t
                    dsc1 = grp_l1_c.create_dataset(dsc_str, (len(x1i),), compression="gzip")
                    dsc1[...] = csi
                    dsc1.attrs["time"] = t
                if dsv_str not in grp_l1_p:
                    grp_l1_p.attrs["t_final"] = t
                    dsv1 = grp_l1_p.create_dataset(dsv_str, (len(x1i),), compression="gzip")
                    dsv1[...] = p1i
                    dsv1.attrs["time"] = t

        if n == (len(dtt) - 1) or debug:
            prog_str = "%s, " % format_time_str(time_s=t)
            if xbar < 1:
                prog_str += "Qb={0:.3g} nm, ".format(xbar * 1e3 * 1e4)
            else:
                prog_str += "Qb={0:.3g} um, ".format(xbar * 1e4)
            xbar_n = np.argmin(abs(x1i - xbar * 2))  # Get index of 2*xbar

            if round(p1i.max()) != round(voltage):
                prog_str += "Vmax={0:.4g} V, ".format(p1i.max())
                if p1i.max() != p1i[0] and round(p1i[0]) != round(voltage):
                    prog_str += "V0={0:.4g} V, ".format(p1i[0])

            if (
                p1i[:xbar_n].mean() != p1i[0]
                and p1i[:xbar_n].mean() != p1i.max()
                and round(p1i[:xbar_n].mean()) != round(voltage)
            ):
                prog_str += "Vb={0:.4g} V, ".format(p1i[:xbar_n].mean())
            if thick != thick_mesh:
                prog_str += "VL={0:.4g} V, ".format(p1i[-1])
            # prog_str += "Eave={0:1.2E} V/cm, ".format((p1i / (thick - x1i))[:-1].mean())
            prog_str += "E0={0:1.2E} V/cm, EL={1:.2E} V/cm, ".format(gp1, gp2)
            C1exp = int(np.log10(c1i[0]))
            C2exp = int(np.log10(c2i[0]))
            Csexp = int(np.log10(csi[0]))
            if C1exp >= Csexp - 10 and C2exp >= Csexp - 10:
                prog_str += "C0[+, 0, sum]=[{0:.2f}, {1:.2f}, {2:.2f}]E+{3:d}, ".format(c1i[0]/10**Csexp, c2i[0]/10**Csexp, csi[0]/10**Csexp, Csexp)
            else:
                prog_str += "C0={0:.2E}, ".format(csi[0])
            if csi[-1] > csi.max() * 1e-10 or csi[-1] < cbulk:
                prog_str += "CL={0:.2E}, ".format(csi[-1])
            prog_str += "Cb={0:.2E}, Ct={1:.2E}".format(
                integrate.trapz(csi[:xbar_n], x1i[:xbar_n] * 1e-4), integrate.trapz(csi, x1i * 1e-4)
            )

            fcallLogger.info(prog_str)

        try:
            if any(csi < 0):
                raise RuntimeError("Invalid Concentration Values")
            # c1i[np.argmax(c1i <= cbulk):] = cbulk
            solve_1G(gp1, gp2, dti, t)
            

            # Update the electric potential gradients
            (
                gp1,
                gp2,
                _,
                _,
                _,
            ) = update_potential_bc(f1_trial_tig, bias=voltage)
            solve_1N(gp1, gp2, dti, t)

            # Update previous solution
            f1_trial_ti0.assign(f1_trial_ti1)
            c2_trial_ti0.assign(c2_trial_ti1)

        except KeyboardInterrupt:
            fcallLogger.error("Program halted")
            x1i, c1i, p1i = get_solution_array1(mesh1, f1_trial_ti0)
            x2i, c2i = get_solution_array2(mesh2, c2_trial_ti0)
            csi = c1i + c2i
            return t_sim, x1i, csi, p1i, c_max

        except RuntimeError:
            message = "Could not solve for time {0:.1f} h. D1 = {1:.3E} cm2/s CSi = {2:.1E} 1/cm^3,\t".format(
                t / 3600, diffusivity, csi[-1] * 1e1
            )
            message += "T = {0:3.1f} °C, E = {1:.1E} MV/cm, tmax: {2:3.2f} hr, XPOINTS = {3:d}, tpoints: {4}".format(
                tempC, voltage / thick * 1e-6, time_s / 3600, xpoints, format_time_str(dt_max)
            )
            fcallLogger.info(message)
            if fcall <= max_calls:
                dt_max = kwargs.pop("dt_max", int(time_s / 500)) / 2

                fcallLogger.info(
                    "Trying with a smaller dt: {0}, refinement step: {1:d}".format(
                        format_time_str(dt_max), fcall
                    )
                )
                fcall += 1

                return double_ion(
                    tempC=tempC,
                    voltage=voltage,
                    thick=thick,
                    time_s=time_s,
                    diffusivity=diffusivity,
                    er=er,
                    material=material,
                    csurf=csurf,
                    cbulk=cbulk,
                    fcall=fcall,
                    dt_max=dt_max,
                    **kwargs,
                )

            else:
                fcallLogger.error("Reached max refinement without success...")
                x1i, c1i, p1i = get_solution_array1(mesh1, f1_trial_ti0)
                x2i, c2i = get_solution_array2(mesh2, c2_trial_ti0)
                csi = c1i + c2i
                return t_sim, x1i * 1e4, csi, p1i, c_max

    if h5fn is not None:
        with h5py.File(h5fn_tmp, "a") as hf:
            hf[material].attrs["Cmax"] = c_max
            hf.close()
        if os.path.exists(h5fn):
            os.remove(h5fn)
        os.rename(h5fn_tmp, h5fn)

        hf_pths = [f for f in f_find(os.sep.join(h5fn.split(os.sep)[:-1]), re_filter=".h5")]
        hf_nms = [f.stem for f in hf_pths]
        # hf_nms, hf_pths = get_filenames(
        #     os.sep.join(h5fn.split(os.sep)[:-1]), file_type=".h5", as_path=False, as_name=False
        # )
        [
            os.remove(hf_pths[n])
            for n, f in enumerate(hf_nms)
            if f.startswith(h5fn.split(os.sep)[-1][:-3] + "_")
        ]
    tm_delta = time.time() - tm_start
    fcallLogger.info("Stop: {0} [{1} elapsed]".format(
        date.now().strftime("%D_%H:%M:%S"),
        format_time_str(tm_delta)
        )
    )
    return t_sim, x1i * 1e4, csi, p1i, c_max
