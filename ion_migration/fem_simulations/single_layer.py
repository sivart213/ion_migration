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
import pnptransport.utils as utils
import sympy as sp
import h5py
import os
from typing import Union
import scipy.constants as constants
from scipy import integrate
from scipy.stats import linregress
from datetime import datetime as date

from utilities import get_filenames

q_red = 1.6021766208  # x 1E-19 C
e0_red = 8.854187817620389  # x 1E-12 C^2 / J m
CM3TOUM3 = 1e-12


def single_layer(
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

        tsim, x1, c_ftrial_ti1, x2, c2, cmax = pnpfs.single_layer_constant_source_flux(
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
        valency: integer
            The valency of the ion
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
    c_ftrial_ti1: np.ndarray
        The final concentration profile as a function of depth in SiNx in cm\ :sup:`-3`\.
    potential: np.ndarray
        The final potential profile as a function of depth in SiNx in V.
    cmax: float
        The maximum concentration in silicon nitride in cm\ :sup:`-3`\.
    """

    # q_red = 1.6021766208  # x 1E-19 C
    # e0_red = 8.854187817620389  # x 1E-12 C^2 / J m
    voltage_pin = kwargs.get("voltage_pin", True)
    thick_mesh = kwargs.get("thick_mesh", thick)
    thick_mesh_ref = kwargs.get("thick_mesh_ref", thick_mesh * 0.1)
    thick_na = kwargs.get("thick_na", 0.0)
    csurf_vol = kwargs.get("csurf_vol", csurf)
    in_flux = kwargs.get("in_flux", ("surface", 1e-4))
    out_flux = kwargs.get("out_flux", ("closed", 0))
    dx_max = kwargs.get("dx_max", 1e-7)
    dx_min = kwargs.get("dx_min", dx_max * 0.01)
    dt_max = kwargs.get("dt_max", int(time_s / 500))
    dt_base = kwargs.get("dt_base", 3)
    max_calls = kwargs.get("max_calls", 5)
    max_iter = kwargs.get("max_iter", 1000)

    valency = kwargs.get("valency", 1.0)
    h5fn = kwargs.get("h5_storage", None)
    debug = kwargs.get("debug", False)
    relax_param = kwargs.get("relax_param", 1.0)
    note = kwargs.get("note", "None")
    func = kwargs.get("func", None)
    screen = kwargs.get("screen", 0)

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

    # Estimate the diffusion coefficients for the given temperature
    tempK = tempC + 273.15

    # Transform everything to um, s
    # D1ums = diffusivity * 1e8

    # The constant mobility z * q * D1 / (kb * T) :: (a.u. * C * cm2/s / (C*V/K * K)) -> cm2/(Vs)
    mob = valency * constants.elementary_charge * diffusivity / (constants.Boltzmann * tempK)
    # mu2 = 0.0
    # The constant z * q / (e0 * er) :: (a.u. * C / (C/Vm * a.u.)) -> Vm * 100 -> Vcm
    qee = valency * constants.elementary_charge / (constants.epsilon_0 * er) * 100

    dlf.set_log_level(50)
    logging.getLogger("FFC").setLevel(logging.WARNING)

    if debug:
        fcallLogger.info("********* Global parameters *********")
        fcallLogger.info("Start: {}".format(date.now().strftime("%D_%H:%M:%S")))
        fcallLogger.info("-------------------------------------")
        fcallLogger.info("Time: {0}".format(utils.format_time_str(time_s)))
        fcallLogger.info("Temperature: {0:.1f} °C".format(tempC))
        if "inter" in out_flux[0].lower():
            fcallLogger.info("Source concentration: {0:.4E} (Na atoms/cm^3)".format(csurf_vol))
        else:
            fcallLogger.info("Source concentration: {0:.4E} (Na atoms/cm^2)".format(csurf))
        fcallLogger.info("*************** {} ******************".format(material))
        fcallLogger.info("Material Thickness: {0:.3G} um".format(thick * 1e4))
        fcallLogger.info("Mesh Thickness: {0:.3G} um".format(thick_mesh * 1e4))
        fcallLogger.info("er: {0:.2f}".format(er))
        fcallLogger.info("Voltage: {0:.1f} V".format(voltage))
        fcallLogger.info("Electric Field: {0:.4E} MV/cm".format(voltage / thick * 1e-6))
        fcallLogger.info("D: {0:.4E} cm^2/s".format(diffusivity))
        fcallLogger.info("Ionic mobility: {0:.4E} um^2/ V*s".format(mob * 1e8))

    # Create classes for defining parts of the boundaries and the interior
    # of the domain

    tol = 1e-12

    class Top(dlf.SubDomain):
        def inside(self, x_, on_boundary):
            return near(x_[0], 0.0, tol) and on_boundary

    class Bottom(dlf.SubDomain):
        def inside(self, x_, on_boundary):
            return near(x_[0], thick_mesh, tol) and on_boundary

    def get_solution_array(mesh, sol):
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

    top = Top()
    bottom = Bottom()

    # Create mesh and define function space
    xpoints = int(thick_mesh / dx_max)
    mesh1 = dlf.IntervalMesh(xpoints, 0.0, thick_mesh)

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

        thick_mesh_ref = thick_mesh_ref / 1.5

    if debug:
        fcallLogger.info("Refined meshes.")
        fcallLogger.info("********** Mesh 1 **********")
        fcallLogger.info("Elements: {0}".format(len(mesh1.coordinates())))
        fcallLogger.info(
            "MIN DX: {0:.3E} um, MAX DX {1:.3E} um".format(mesh1.hmin() * 1e4, mesh1.hmax() * 1e4)
        )
        fcallLogger.info("DX Ranges: {0}".format(", ".join(nor_ranges)))

    # Initialize mesh function for boundary domains
    boundaries1 = dlf.MeshFunction("size_t", mesh1, mesh1.topology().dim() - 1)
    boundaries1.set_all(0)

    top.mark(boundaries1, 1)
    bottom.mark(boundaries1, 2)

    # Define the measures
    ds1 = dlf.Measure("ds", domain=mesh1, subdomain_data=boundaries1)
    dx1 = dlf.Measure("dx", domain=mesh1, subdomain_data=boundaries1)

    # Generate the initial profile
    f = sp.symbols("f", cls=sp.Function)
    x = sp.Symbol("x[0]", real=True)

    ode_vbias = sp.Eq(f(x).diff(x, x), 0)
    f_vbias = sp.dsolve(ode_vbias, f(x), simplify=False, ics={f(0): voltage, f(thick): 0})

    if thick_na == 0:
        conc = sp.Piecewise(
            (0, x < 0),
            (csurf, x <= mesh1.hmin()*2),  # CM3TOUM3
            (cbulk, x <= thick),  # CM3TOUM3
            (0, x > thick),
            evaluate=False,
        )

        ion_dens = sp.Piecewise(
            (0, x < 0),
            (qee * csurf, x <= mesh1.hmin()*2),  # CM3TOUM3
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

        ion_dens = sp.Piecewise(
            (0, x < 0),
            (qee * csurf_vol, x <= thick_na),  # CM3TOUM3
            (qee * cbulk, x <= thick),  # CM3TOUM3
            (0, x > thick),
            evaluate=False,
        )

    ode_v = sp.Eq(f(x).diff(x, x), -1 * ion_dens)

    eion_s = sp.integrate(x / thick * conc, (x, 0, thick)) * -1 * qee

    f_vs = sp.dsolve(
        ode_v,
        f(x),
        simplify=False,
        ics={f(x).diff(x).subs(x, thick): float((f_vbias.rhs.diff(x) + eion_s)), f(thick): 0},
    )

    ftrial_t0 = dlf.Expression(
        (sp.printing.ccode(conc.evalf()), sp.printing.ccode(f_vs.rhs)), degree=1
    )

    # Defining the mixed function space
    CG1 = dlf.FiniteElement("CG", mesh1.ufl_cell(), 1)
    W_elem = dlf.MixedElement([CG1, CG1])
    W = dlf.FunctionSpace(mesh1, W_elem)

    # Defining the "Trial" functions
    ftrial_ti1 = dlf.interpolate(ftrial_t0, W)  # For time i+1
    c_ftrial_ti1, phi_ftrial_ti1 = split(ftrial_ti1)
    ftrial_tig = dlf.interpolate(ftrial_t0, W)  # For time i+1/2
    c_ftrial_tig, phi_ftrial_tig = split(ftrial_tig)
    ftrial_ti0 = dlf.interpolate(ftrial_t0, W)  # For time i
    c_ftrial_ti0, phi_ftrial_ti0 = split(ftrial_ti0)

    # Define the test functions
    ftest = dlf.TestFunction(W)
    (c_ftest, phi_ftest) = split(ftest)

    du1 = dlf.TrialFunction(W)

    ftrial_ti1.set_allow_extrapolation(True)
    ftrial_tig.set_allow_extrapolation(True)
    ftrial_ti0.set_allow_extrapolation(True)

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
        bcs_ = [dlf.DirichletBC(W.sub(1), bias_L, boundaries1, 2)]
        if pin:
            bcs_.insert(0, dlf.DirichletBC(W.sub(1), bias_0, boundaries1, 1))

        if "const" in in_flux[0].lower():
            bcs_.insert(0, dlf.DirichletBC(W.sub(0), csurf_vol, boundaries1, 1))
        return bcs_

    volt_mesh0 = (1 - thick_mesh / thick) * voltage
    bcs = update_bcs(voltage, volt_mesh0, voltage_pin)

    thick_flux = diffusivity / mesh1.hmin() / 10

    def get_variational_form(c_ftrial, phi_ftrial, gp1_, gp2_, time_i):
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

        integ_xi = dlf.Expression("exp(kd*x[0])", kd=screen, degree=1)

        a = -diffusivity * inner(grad(c_ftrial), grad(c_ftest)) * dx1  # Anp term 1
        a += c_grad_01 * c_ftest * ds1(1)  # Anp term2 in; i.e. bc in
        a += c_grad_12 * c_ftest * ds1(2)  # Anp term2 out; i.e. bc out
        a -= mob * c_ftrial * inner(grad(phi_ftrial), grad(c_ftest)) * dx1  # Anp term 3
        a += mob * gp1_ * c_ftrial * c_ftest * ds1(1)  # Anp term 4 in
        a += mob * gp2_ * c_ftrial * c_ftest * ds1(2)  # Anp term 4 out

        a -= inner(grad(phi_ftrial), grad(phi_ftest)) * dx1  # Ap term 1
        a += qee * c_ftrial * phi_ftest * integ_xi * dx1  # Anp term 3
        a += gp1_ * phi_ftest * ds1(1)  # Anp term 2 in
        a += gp2_ * phi_ftest * ds1(2)  # Anp term 2 out

        return a

    def getTRBDF2ta(c_ftrial, phi_ftrial):
        integ_xi = dlf.Expression("exp(kd*x[0])", kd=screen, degree=1)
        r = (
            diffusivity * div(grad(c_ftrial))
            + div(grad(phi_ftrial))
            + mob * c_ftrial * div(grad(phi_ftrial))
            + mob * inner(grad(phi_ftrial), grad(c_ftrial))
            + qee * c_ftrial * integ_xi
        )
        return r

    def update_potential_bc(uui, bias: float = voltage):
        # The total concentration in the oxide (um-2)
        #        c_ftrial,phi_ftrial = ui.split()
        #        Ctot = assemble(c_ftrial*dx)
        # The integral in Poisson's equation solution (Nicollian & Brews p.426)
        # units: 1/um
        #        Cint = assemble(c_ftrial*dlf.Expression('x[0]',degree=1)*dx)

        # Get the solution in an array form
        uxi, uci, upi = get_solution_array(mesh1, uui)

        # The integrated concentration in the oxide (cm-2) <- Check: (1/cm^3) x (um) x (1E-4 cm/1um)
        Ctot_ = integrate.simps(uci * np.exp(screen * uxi), uxi)  # * 1e-4
        # The integral in Poisson's equation solution (Nicollian & Brews p.426)
        # units: 1/cm <------------- Check: (1/cm^3) x (um^2) x (1E-4 cm/1um)^2
        Cint_ = integrate.simps(uxi * uci * np.exp(screen * uxi), uxi)  # * 1e-8
        # The centroid of the charge distribution
        xbar_ = Cint_ / Ctot_  # * 1e4

        # The surface charge density at silicon C/cm2
        scd_si = -constants.e * (xbar_ / thick) * Ctot_
        # scd_si2 = -constants.e * integrate.simps( uxi / thick * uci * np.exp(screen * uxi), uxi)
        # The surface charge density at the gate C/cm2
        scd_g = -constants.e * (1.0 - xbar_ / thick) * Ctot_
        # scd_g2 = constants.e * integrate.simps((uxi - thick) / thick * uci * np.exp(screen * uxi), uxi)

        # The applied electric field in V/cm
        field_app = bias / thick
        # The electric field at the gate interface V/um
        # (C / cm^2) * (J * m / C^2 ) x ( 1E2 cm / 1 m) x ( 1E cm / 1E4 um)
        gp1_ = field_app + scd_g / (constants.epsilon_0 * er) * 100  # x
        # gp1_2 = field_app + scd_g2 / (constants.epsilon_0 * er) * 100  # x
        # The electric field at the Si interface V/um
        gp2_ = -(field_app - scd_si / (constants.epsilon_0 * er) * 100)  # x
        # gp2_2 = -(field_app - scd_si2 / (constants.epsilon_0 * er) * 100)  # x

        # Since grad(phi) = -field
        # s1: <-
        # s2: ->
        #        gp1 = field_source
        # gp1_ = field_source_
        # gp2_ = -field_sink_

        return gp1_, gp2_, Cint_, Ctot_, xbar_

    hk1 = dlf.CellDiameter(mesh1)

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

    def get_solvers_1(gp1_, gp2_, dt_, time_i, xxi):
        a10 = get_variational_form(c_ftrial_ti0, phi_ftrial_ti0, gp1_, gp2_, time_i)
        a1G = get_variational_form(c_ftrial_tig, phi_ftrial_tig, gp1_, gp2_, time_i)
        a11 = get_variational_form(c_ftrial_ti1, phi_ftrial_ti1, gp1_, gp2_, time_i)

        F1G = (1.0 / dt_) * (c_ftrial_tig - c_ftrial_ti0) * c_ftest * dx1 - TRF * (a1G + a10)
        F1N = (1.0 / dt_) * (
            c_ftrial_ti1 - BDF2_T1 * c_ftrial_tig + BDF2_T2 * c_ftrial_ti0
        ) * c_ftest * dx1 - BDF2_T3 * a11

        # SUPG stabilization
        b1_ = mob * dlf.Dx(phi_ftrial_ti0, 0)
        nb1 = sqrt(dot(b1_, b1_) + dlf.DOLFIN_EPS)
        Pek1 = nb1 * hk1 / (2.0 * diffusivity)

        b2_ = mob * dlf.Dx(phi_ftrial_tig, 0)
        nb2 = sqrt(dot(b2_, b2_) + dlf.DOLFIN_EPS)
        Pek2 = nb2 * hk1 / (2.0 * diffusivity)

        tau1 = dlf.conditional(
            gt(Pek1, dlf.DOLFIN_EPS),
            (hk1 / (2.0 * nb1))
            * (((exp(2.0 * Pek1) + 1.0) / (exp(2.0 * Pek1) - 1.0)) - 1.0 / Pek1),
            0.0,
        )
        tau2 = dlf.conditional(
            gt(Pek2, dlf.DOLFIN_EPS),
            (hk1 / (2.0 * nb2))
            * (((exp(2.0 * Pek2) + 1.0) / (exp(2.0 * Pek2) - 1.0)) - 1.0 / Pek2),
            0.0,
        )

        #  get the skew symmetric part of the L operator
        # LSSNP = dot(vel2,dlf.Dx(v2,0))
        Lss1 = (
            mob * inner(grad(phi_ftrial_tig), grad(c_ftest))
            + (mob / 2) * div(grad(phi_ftrial_tig)) * c_ftest
        )
        Lss2 = (
            mob * inner(grad(phi_ftrial_ti1), grad(c_ftest))
            + (mob / 2) * div(grad(phi_ftrial_ti1)) * c_ftest
        )
        # SUPG Stabilization term
        ta = getTRBDF2ta(c_ftrial_tig, phi_ftrial_tig)
        tb = getTRBDF2ta(c_ftrial_ti0, phi_ftrial_ti0)
        tc = getTRBDF2ta(c_ftrial_ti1, phi_ftrial_ti1)
        ra = inner(((1 / dt_) * (c_ftrial_tig - c_ftrial_ti0) - TRF * (ta + tb)), tau1 * Lss1) * dx1
        rb = (
            inner(
                (
                    c_ftrial_ti1 / dt_
                    - BDF2_T1 * c_ftrial_tig / dt_
                    + BDF2_T2 * c_ftrial_ti0 / dt_
                    - BDF2_T3 * tc
                ),
                tau2 * Lss2,
            )
            * dx1
        )

        F1G += ra
        F1N += rb

        J1G = dlf.derivative(F1G, ftrial_tig, du1)
        J1N = dlf.derivative(F1N, ftrial_ti1, du1)  # J1G

        problem1N = dlf.NonlinearVariationalProblem(
            F1N, ftrial_ti1, bcs, J1N, form_compiler_parameters=ffc_options
        )
        problem1G = dlf.NonlinearVariationalProblem(
            F1G, ftrial_tig, bcs, J1G, form_compiler_parameters=ffc_options
        )
        solver1N_ = dlf.NonlinearVariationalSolver(problem1N)
        solver1N_.parameters.update(newton_solver_parameters)
        solver1G_ = dlf.NonlinearVariationalSolver(problem1G)
        solver1G_.parameters.update(newton_solver_parameters)

        # Return the solvers and the variational forms to get an estimate of the truncation error
        return solver1N_, solver1G_, a10, a1G, a11

    dt = dt_max
    num_t = 50
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
    
    # dtt = np.concatenate([t1[0], dtt, [dt]])
    size_n = len(dtt)
    del dt, num_t, t1, t2

    if debug:
        fcallLogger.info("**** Time stepping *****")
        fcallLogger.info(
            "Min dt: {0}, Max dt: {1}.".format(
                utils.format_time_str(np.amin(dtt)), utils.format_time_str(np.amax(dtt))
            )
        )
        fcallLogger.info("Simulation time: {0}.".format(utils.format_time_str(time_s)))
        fcallLogger.info("Number of time steps: {0}".format(len(t_sim)))

    x1i, c1i, p1i = get_solution_array(mesh1, ftrial_ti0)
    c_max = -np.inf

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
            grp_l1.attrs["er"] = er
            grp_l1.attrs["T"] = tempC
            grp_l1.attrs["V"] = voltage
            grp_l1.attrs["E"] = voltage / thick * 1e-6

            for kw, va in kwargs.items():
                if kw in ["h5_storage", "relax_param", "func", "debug", "fcallLogger"]:
                    continue
                if isinstance(va, tuple) and isinstance(va[0], str):
                    grp_l1.attrs[f"{kw}_{va[0]}"] = va[1]
                elif "screen" in kw.lower() and va != 0:
                    grp_l1.attrs[kw] = -1 / va * 1e7
                else:
                    grp_l1.attrs[kw] = va

            grp_l1.create_group("concentration")
            grp_l1.create_group("potential")

    """
    This section starts the integration loop
    """
    if debug:
        fcallLogger.info("Starting time integration loop...")

    for n, t in enumerate(t_sim):
        dti = dtt[n]

        volt_intf = 0.0
        if thick != thick_mesh:
            volt_intf = mid_bias(x1i, p1i, [thick, 0.0], thick_mesh)

        if p1i[0] < voltage:
            bcs = update_bcs(voltage, volt_intf, True)
        else:
            bcs = update_bcs(voltage, volt_intf, voltage_pin)

        gp1, gp2, Cint, Ctot, xbar = update_potential_bc(ftrial_ti0, bias=voltage)
        x1i, c1i, p1i = get_solution_array(mesh1, ftrial_ti0)

        c_max = max(c_max, np.amax(c1i))
        thick_flux = thick_flux * c1i[-1] / cbulk
        if func is not None:
            func(x1i * 1e4, c1i, p1i)
        # scatter(pd.DataFrame([x1i,c1i,p1i], index=["depth", "conc", "volt"]), x="")
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
                    dsc1[...] = c1i
                    dsc1.attrs["time"] = t
                if dsv_str not in grp_l1_p:
                    grp_l1_p.attrs["t_final"] = t
                    dsv1 = grp_l1_p.create_dataset(dsv_str, (len(x1i),), compression="gzip")
                    dsv1[...] = p1i
                    dsv1.attrs["time"] = t

        if n == (size_n - 1) or debug:
            prog_str = "%s, " % utils.format_time_str(time_s=t)
            if xbar < 1:
                prog_str += "Qb={0:.3g} nm, ".format(xbar * 1e3 * 1e4)
            else:
                prog_str += "Qb={0:.3g} um, ".format(xbar * 1e4)
            xbar_n = np.argmin(abs(x1i - xbar * 2))  # Get index of 2*xbar
            # vpin = p1i[0]
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
            prog_str += "C0={0:.2E}, ".format(c1i[0])
            if c1i[-1] > c1i.max() * 1e-10 or c1i[-1] < cbulk:
                prog_str += "CL={0:.2E}, ".format(c1i[-1])
            prog_str += "Cb={0:.2E}, Ct={1:.2E}".format(
                integrate.trapz(c1i[:xbar_n], x1i[:xbar_n] * 1e-4), integrate.trapz(c1i, x1i * 1e-4)
            )

            fcallLogger.info(prog_str)

        try:
            if any(c1i < 0):
                raise RuntimeError("Invalid Concentration Values")
            # c1i[np.argmax(c1i <= cbulk):] = cbulk
            solver1N, solver1G, f_00, f_01, f_02 = get_solvers_1(gp1, gp2, dti, t, x1i)
            solver1G.solve()

            # Update the electric potential gradients
            (
                gp1,
                gp2,
                _,
                _,
                _,
            ) = update_potential_bc(ftrial_tig, bias=voltage)
            solver1N, solver1G, _, _, _ = get_solvers_1(gp1, gp2, dti, t, x1i)

            solver1N.solve()
            # Update previous solution
            ftrial_ti0.assign(ftrial_ti1)

        except KeyboardInterrupt:
            fcallLogger.error("Program halted")
            x1i, c1i, p1i = get_solution_array(mesh1, ftrial_ti0)
            return t_sim, x1i, c1i, p1i, c_max

        except RuntimeError:
            message = "Could not solve for time {0:.1f} h. D1 = {1:.3E} cm2/s CSi = {2:.1E} 1/cm^3,\t".format(
                t / 3600, diffusivity, c1i[-1] * 1e1
            )
            message += "T = {0:3.1f} °C, E = {1:.1E} MV/cm, tmax: {2:3.2f} hr, XPOINTS = {3:d}, tpoints: {4}".format(
                tempC, voltage / thick * 1e-6, time_s / 3600, xpoints, utils.format_time_str(dt_max)
            )
            fcallLogger.info(message)
            if fcall <= max_calls:
                dt_max = kwargs.pop("dt_max", int(time_s / 500)) / 2

                fcallLogger.info(
                    "Trying with a smaller dt: {0}, refinement step: {1:d}".format(
                        utils.format_time_str(dt_max), fcall
                    )
                )
                fcall += 1

                return single_layer(
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
                x1i, c1i, p1i = get_solution_array(mesh1, ftrial_ti0)
                return t_sim, x1i * 1e4, c1i, p1i, c_max

    if h5fn is not None:
        with h5py.File(h5fn_tmp, "a") as hf:
            hf[material].attrs["Cmax"] = c_max
            hf.close()
        if os.path.exists(h5fn):
            os.remove(h5fn)
        os.rename(h5fn_tmp, h5fn)
        hf_nms, hf_pths = get_filenames(
            os.sep.join(h5fn.split(os.sep)[:-1]), file_type=".h5", as_path=False, as_name=False
        )
        [
            os.remove(hf_pths[n])
            for n, f in enumerate(hf_nms)
            if f.startswith(h5fn.split(os.sep)[-1][:-3] + "_")
        ]

    return t_sim, x1i * 1e4, c1i, p1i, c_max
