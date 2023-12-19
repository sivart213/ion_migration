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
test edit
"""

# %% Imports
from __future__ import annotations
import re
import numpy as np
import sympy as sp
import sympy.physics.units as su
from abc import ABC, abstractmethod
# from collections import UserDict
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, _FIELDS, _FIELD, InitVar # astuple, asdict
import scipy.constants as constants
from scipy import integrate
from scipy.stats import linregress
from research_tools.functions import get_const, map_plt, convert_val, get_const, all_symbols, has_units
from research_tools.equations import screened_permitivity, debye_length

# %% Base Class
class BaseClass(object):
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, val):
        setattr(self, key, val)
    
    def update(self, *args, **kwargs):
        for k, v in dict(args).items():
            if k in self.__dict__.keys():
                self[k] = v
        for k, v in kwargs.items():
            (kwargs.pop(k) for k in kwargs.keys() if k not in self.__dict__.keys())
            self[k] = v
        return self
    
    def copy(self, **kwargs):
        """Return a new instance copy of obj"""
        (kwargs.pop(k) for k in kwargs.keys() if k not in self.__dict__.keys())
        kwargs = {**self.__dict__, **kwargs}
        return self.__class__(**kwargs)
    
    def inputs(self):
        return list(self.__dict__.keys())
    
    def sanitize(self, raw, key_names=None, create_instance=False, **init_kwargs):
        """
        dicta would be the old kwargs (key: value)
        dictb would be the renaming dict (old K: new K)
        dictc would be the cls input args
        """
        if isinstance(key_names, (list, tuple)):
            try:
                key_names = {k: v for k, v in key_names}
            except ValueError:
                pass
        if isinstance(key_names, dict):
            raw = {key_names.get(k, k): v for k, v in raw.items()}
        
        kwargs = {k: raw.get(k, v) for k, v in self.__dict__.items()}
        if create_instance:
            return self.__class__(**init_kwargs, **kwargs)
        return kwargs

class DictMixin(Mapping, BaseClass):
    def __iter__(self):
        return (f.name for f in fields(self))

    def __getitem__(self, key):
        if not isinstance(key, str):
            return [self[k] for k in key]
        field = getattr(self, _FIELDS)[key]
        if field._field_type is not _FIELD:
            raise KeyError(f"'{key}' is not a dataclass field.")
        return getattr(self, field.name)
    
    def __setitem__(self, key, val):
        setattr(self, key, val)
    
    def __len__(self):
        return len(fields(self))

# %% Ion classes
@dataclass()
class Ion(DictMixin):
    """Return sum of squared errors (pred vs actual)."""
    D: float = 1.0e-10
    z: float = 1.0
    T: float = 298.15
  
    qz:float = field(init=False)
    kbT:float = field(init=False)
    mob:float = field(init=False)
    
    CtoK:InitVar[bool] = False
    
    def __post_init__(self, CtoK):
        """Return sum of squared errors (pred vs actual)."""
        if CtoK:
            self.convert_T()
                
    @property
    def qz(self)->float:
        """Return q*z. Units: Coulomb"""
        return self.z * constants.e

    @qz.setter
    def qz(self, _): pass

    @property
    def kbT(self)->float:
        """Return k_B*T given e_r. Units: Coulomb*V"""
        return constants.Boltzmann * self.T

    @kbT.setter
    def kbT(self, _): pass 
    
    @property
    def mob(self)->float:
        """Return z*q*D/(kb*T) given e_r. Units: cm2/(Vs)"""
        return self.qz * self.D / self.kbT

    @mob.setter
    def mob(self, _): pass

    def qz_e(self, er):
        """Return q*z/(e_r*e_0) given e_r. Units: Vcm"""
        return self.qz / (constants.epsilon_0 * er / 100)

    def convert_T(self, to_unit="k", update=True):
        """Convert T to requested unit: K from C or vice versa"""
        res = self.T
        if to_unit.lower() == "c":
            res = self.T - 273.15
        elif self.T != 298.15:
            res = self.T + 273.15
        
        if update:
            self.T = res
            return self
        else:
            return res
    

@dataclass()
class IonConcentration(DictMixin):
    """Return sum of squared errors (pred vs actual)."""
    C_s: float = 0.0  # Source
    C_eq: float = 0.0  # Equalibrium
    C_0: float = 0.0  # Initial
    S_0: float = 0.0  # Source in cm-2
    C_b: float = 1e-20    # background
    C: float = 0.0  # Current
    C_out: float = 0.0  # Current C of next layer
    
    L_j0: float = 0.0  # Initial ion depth

    
    def __post_init__(self):
        """Return sum of squared errors (pred vs actual)."""
        if isinstance(self.L_j0, (list, tuple, np.ndarray)):
            self.L_from_dx(*self.L_j0)
        
        self.C_L_relations(None, None, None)
        c_init = self.C
        if isinstance(c_init, (tuple, list, np.ndarray)):
            c_init = sum(self.C)
        c_arr = np.array([self.C_s, self.C_eq, self.C_0, c_init, self.S_0])
        if len(np.nonzero(c_arr)[0]) != len(c_arr):
            c_ind = np.nonzero(c_arr)[0]
            if 1 not in c_ind:
                self.C_eq = self.C_s
            if 2 not in c_ind and 3 in c_ind:
                self.C_0 = c_init
            else:
                self.C_0 = self.C_s
            if 3 not in c_ind and 2 in c_ind:
                self.C = self.C_0
            else:
                self.C = self.C_s
    
    def L_from_dx(self, *dx, update=True):
        """Calculate L from dx segments"""
        res  = max(dx)/2**int(np.log(max(dx) / min(dx)) / np.log(2))*2
        if update:
            self.L_j0 = res
            return self
        else:
            return res
    
    def C_L_relations(self, Cs=0, S0=0, L=0, update=True):
        if isinstance(L, (list, tuple, np.ndarray)):
            L = self.L_from_dx(*L, update=False)

        L = self.L_j0 if L is None else L
        Cs = self.C_s if Cs is None else Cs
        S0 = self.S_0 if S0 is None else S0
        
        Cs = max(self.C_eq, self.C_0) if Cs == 0 and S0 == 0 else Cs
        Cs = 1e10 if Cs == 0 and S0 == 0 else Cs
        
        if L != 0 and S0 == 0:
            S0 = Cs * L
        elif L != 0:
            Cs = S0 / L
        elif Cs != 0 and S0 != 0:
            L = S0 / Cs
        else:
            L = 2e-9
            S0 = Cs * L
        
        if update:
            self.C_s = Cs
            self.S_0 = S0
            self.L_j0 = L
            return self
        else:
            return Cs, S0, L

    # def calc_screen(self, er, update=True):
    #     """Convert T to K from C or vice versa"""
    #     epsilon = constants.epsilon_0 * er / 100
    #     res = np.sqrt(epsilon*self.kbT/(self.qz**2*self.C_eq))*1e7
    #     if update:
    #         self.screen = res
    #         return self
    #     else:
    #         return res
# %% External Functions
def update_potential_bc(uxi, uci, upi, ion, screen, bias, C):
    """
    Generate voltage bc
    """
    
    if isinstance(screen, str):
        scr_exp = np.exp(-1 * ion.qz * upi / ion.kbT)
    elif screen is None or screen == 0:
        scr_exp = 1
    else:
        scr_exp = np.exp(-1 * abs(1/(screen*1e-7)) * uxi)

    Ctot_ = integrate.simps(uci * scr_exp, uxi)  # * 1e-4
  
    Cint_ = integrate.simps(uxi * uci * scr_exp, uxi)  # * 1e-8
    
    xbar_ = uxi.mean()
    if Ctot_ != 0:
        xbar_ = Cint_ / Ctot_  # * 1e4

    # The surface charge density at silicon C/cm2
    scd_si = -1 * ion.qz * (xbar_ / C.L) * Ctot_
   
    # The surface charge density at the gate C/cm2
    scd_g = -1 * ion.qz * (1.0 - xbar_ / C.L) * Ctot_
   
    field_app = bias / C.L
  
    gp1_ = field_app + scd_g / (C.e) * 100  # x
    # The electric field at the Si interface V/um
    gp2_ = -(field_app - scd_si / (C.e) * 100)  # x

    return gp1_, gp2_, Cint_, Ctot_, xbar_


def make_piecewise(*args, var="x", **kwargs):
    args=tuple(args)
    if not isinstance(args[0], (list, tuple)):
        args = tuple([args])
    elif isinstance(args[0][0], (list, tuple)):
        args = tuple(args[0])

    if len(args) == 1 and len(args[0]) == 1:
        return args[0][0]

    if not isinstance(args[-1], (tuple, list)):
        args = args[:-1] + tuple([(args[-1], True)])
    elif len(args[-1]) == 1:
        # args[-1] = (args[-1][0], True)
        args = args[:-1] + tuple([(args[-1][0], True)])
    elif not isinstance(args[-1][-1], bool):
        args = args+tuple([(0, True)])

    if isinstance(var, str):
        var = sp.Symbol(var, real=True)
    pairs = []
    for a in args:
        if len(a) > 2:
            pairs.append((a[0], sp.Interval(*a[1:]).contains(var)))
        elif isinstance(a[1], (tuple, list)):
            pairs.append((a[0], sp.Interval(*a[1]).contains(var)))
        elif isinstance(a[1], bool):
            pairs.append(a)
        elif not isinstance(a[1], str):
            pairs.append((a[0], var < a[1]))
        else:
            a1 = re.search("[<>=]+", a[1])
            var1 = re.search(str(var), a[1])
            kvars = kwargs.get("kwargs", kwargs)
            kvars[str(var)] = var
            if not var1:
                if not a1:
                    expr = sp.parse_expr(str(var)+"<"+a[1])
                elif a1.start() == 0:
                    expr = sp.parse_expr(str(var)+a[1])
                elif a1.end() == len(a[1]):
                    expr = sp.parse_expr(a[1]+str(var))
                else:
                    expr = sp.parse_expr(str(var)+"*"+a[1])
            else:
                expr = sp.parse_expr(a[1])
            pairs.append((a[0], expr.subs(kvars)))

    return sp.Piecewise(*[(a[0], a[1]) for a in pairs], evaluate=False)

def poisson_rhs(C, z, epsilon_r):
    if isinstance(C, (tuple, list)):
        return [poisson_rhs(var, z, epsilon_r) for var in C]

    arg_in = vars().copy()

    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    e0 = get_const("e0", *([True] if symbolic else [w_units, ["farad", "cm"]]))

    res = q * z * C / (epsilon_r * e0)
    return res


def poisson_ode(C, z, epsilon_r, f=None, x=None, deg=2):
    arg_in = vars().copy()

    if f is None:
        f = sp.symbols("f", cls=sp.Function)
    if x is None:
        x = []
        if hasattr(C, "free_symbols"):
            x = [var for var in C.free_symbols if "x" in str(var)]
        x = x[0] if len(x) >= 1 else sp.Symbol("x", real=True)

    ion = poisson_rhs(C, z, epsilon_r)
    if deg > 1:
        ion=-ion
    if isinstance(C, sp.Piecewise):
        ion = sp.piecewise_fold(ion)
    return sp.Eq(f(x).diff(*[x]*deg), ion)




# %%  Factories
class AbstractSettings(ABC):
    """
    This is the abstract factory building the migration functions
    """
    
    @abstractmethod
    def flux_in(self):
        pass
    
    @abstractmethod
    def flux_out(self):
        pass
    
    @abstractmethod
    def flux_out(self):
        pass
    
    @abstractmethod
    def variational_form(self):
        pass



class BaseSettings(AbstractSettings):
    """
    Concrete Factories produce a family of products that belong to a single
    variant. The factory guarantees that resulting products are compatible. Note
    that signatures of the Concrete Factory's methods return an abstract
    product, while inside the method a concrete product is instantiated.
    """
    
    def flux_in(self):
        return Flux0()
    
    def flux_out(self):
        return Flux0()
    
    def variational_form(self):
        return linear_diffusion()

class const_flux_w_bias(BaseSettings):
    """
    Each Concrete Factory has a corresponding product variant.
    """

    def flux_in(self):
        return FluxConst()

    def variational_form(self):
        return linear_NP()



# %% Flux Products
class AbstractFlux(ABC, BaseClass):
    @abstractmethod
    def val(self):
        pass
    
    # TODO add method for parsing 2 layer Conc classes plus rate

class Flux0(AbstractFlux):
    def val(self, **kwargs):
        return 0

class FluxConst(AbstractFlux):
    def val(self, C_0, **kwargs):
        return C_0
    
    
class FluxRate(AbstractFlux):
    def val(self, C_0, rate, **kwargs):
        return rate * C_0

    
class FluxInterface(AbstractFlux):
    def val(self, C_0, C_1, rate, m):
        return rate * (C_0 - C_1 / m)



# %% Screening Products
class Screening(ABC, BaseClass):
    @property
    @abstractmethod
    def length(self):
        pass
    
    @abstractmethod
    def exp_arr(self):
        pass
    
    @abstractmethod
    def exp_fem(self):
        pass
    
class Base(Screening):
    @property
    def length(self):
        if not hasattr(self, "_length"):
            self._length = 0
        return self._length
    
    def exp_arr(self, val, **kwargs):
        self._length = kwargs.get("length", self.length)
        return 1
    
    def exp_fem(self, val, **kwargs):
        self._length = kwargs.get("length", self.length)
        return 1

class ScrConst(Screening):
    @property
    def length(self):
        if not hasattr(self, "_length"):
            self._length = 0
        return self._length
    
    def exp_arr(self, val, **kwargs):
        self._length = kwargs.get("length", self.length)
        return np.exp(-1 * abs(1/(self.length*1e-7)) * val)
    
    def exp_fem(self, val, expr, **kwargs):
        self._length = kwargs.get("length", self.length)
        return expr("exp(-1*kd*x[0])", kd=abs(1/(self.length*1e-7)), degree=1)

class ScrCalc(Screening):
    @property
    def length(self):
        if not hasattr(self, "_length"):
            self._length = 0
        return self._length
    
    def exp_arr(self, val, er, ion, C, **kwargs):
        if self.length == 0:
            epsilon = constants.epsilon_0 * er / 100
            self._length = np.sqrt(epsilon*ion.kbT/(ion.qz**2*C.C_eq))*1e7
        if self.length == 0:
            print("need to calc screen length!")
        return np.exp(-1 * abs(1/(self.length*1e-7)) * val)
    
    def exp_fem(self, val, expr, er, ion, C, **kwargs):
        if self.length ==  0:
            epsilon = constants.epsilon_0 * er / 100
            self._length = np.sqrt(epsilon*ion.kbT/(ion.qz**2*C.C_eq))*1e7
        if self.length == 0:
            print("need to calc screen length!")
        return expr("exp(-1*kd*x[0])", kd=abs(1/(self.length*1e-7)), degree=1)


class ScrVolt(Screening):
    @property
    def length(self):
        if not hasattr(self, "_length"):
            self._length = 0
        return self._length
    
    def exp_arr(self, val, ion, **kwargs):
        self._length = kwargs.get("length", self.length)
        return np.exp(-1 * ion.qz * val / ion.kbT)
    
    def exp_fem(self, val, ion, exp, **kwargs):
        self._length = kwargs.get("length", self.length)
        return exp(-1 * ion.qz * val / ion.kbT)
    

# %% Variational form Products
class AbstractForm(ABC, BaseClass):
    @abstractmethod
    def form(self):
        pass
    
    @abstractmethod
    def get_array(self):
        pass
    
    @abstractmethod
    def TRBDF2ta(self):
        pass

class linear_diffusion(AbstractForm):
    """This is diffusion only"""
    def form(self, a):
        print(f"Takes flux_in&out (w cftrial), mob*cftrial*E_in&out, cftest, dx, and ds")
        print(f"Returns Anp terms 1 & 2")
        return 5*a


class linear_NP(AbstractForm):
    """This is the NP where poisson = 0"""
    def form(self, a):
        res = linear_diffusion().func(a)+6/a
        print(f"Additionally takes phiftrial and phiftest")
        print(f"To include Anp terms 3 & 4 & Ap terms 1 & 2")
        return res
     

class linear_NP_alt(AbstractForm):
    """This is the NP wher poisson is ignored, may not work"""
    def form(self, a):
        res = linear_diffusion().func(a)+6/a
        print(f"Additionally takes phiftrial")
        print(f"To include Anp terms 3 & 4")
        return res
    

class linear_PNP(AbstractForm):
    """This is the normal full variation"""

    def form(self, a):
        res = linear_NP().func(a)+a**2
        print(f"Additionally takes and screening exp")
        print(f"To include Ap terms 3")
        return res
    

# %% Mesh Products
class MeshAbstract(ABC, BaseClass):
    @abstractmethod
    def val(self):
        pass
    
    # # DO NOT DELETE, hidden for testing
    @staticmethod
    def get_dx_arr(mesh, dx_min, dx_max, L_refine_0, L_refine_1, debug, dlf):
        nor = int(np.log(dx_max / dx_min) / np.log(2))
        for i in range(nor):
            cell_markers = dlf.MeshFunction("bool", mesh, mesh.topology().dim(), False)
            for cell in dlf.cells(mesh):
                p = cell.midpoint()
                if p[0] <= L_refine_0 or p[0] >= L_refine_1:
                    cell_markers[cell] = True
            mesh = dlf.refine(mesh, cell_markers)
    
            L_refine_0 = L_refine_0 / 1.5
            L_refine_1 = L_refine_1 / 1.5
        return mesh
    
    @staticmethod
    def get_dt_arr(dt_max, dt_base, time_s):
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

        return dtt
    
    @staticmethod
    def SUPG_terms(trial_t0, trial_t1, c_ftest, mesh, ion, dlf):
        b_ = ion.mob * dlf.Dx(trial_t0, 0)
        nb_ = dlf.sqrt(dlf.dot(b_, b_) + dlf.DOLFIN_EPS)
        Pek = nb_ * dlf.CellDiameter(mesh) / (2.0 * ion.D)

        tau_ = dlf.conditional(
            dlf.gt(Pek, dlf.DOLFIN_EPS),
            (dlf.CellDiameter(mesh) / (2.0 * nb_))
            * (((dlf.exp(2.0 * Pek) + 1.0) / (dlf.exp(2.0 * Pek) - 1.0)) - 1.0 / Pek),
            0.0,
        )

        Lss_ = (
            ion.mob * dlf.inner(dlf.grad(trial_t1), dlf.grad(c_ftest))
            + (ion.mob / 2) * dlf.div(dlf.grad(trial_t1)) * c_ftest
        )
        
        return tau_, Lss_

class MeshFinite(AbstractFlux):
    def get_array(self, mesh, sol):
        print("takes mesh, sol, returns x & c arrays")
        xu = mesh.coordinates()
        cu = sol.compute_vertex_values(mesh)
        xyz = np.array(
            [(xu[j], cu[j]) for j in range(len(xu))],
            dtype=[("x", "d"), ("c", "d")],
        )
        xyz.sort(order="x")
        return xyz["x"], xyz["c"]
    
    def TRBDF2ta(self, **kwargs):
        return 0
    
    def func_space(self, **kwargs):
        return 1

class MeshMixed(AbstractFlux):
    def get_array(self, mesh, sol):
        print("takes mesh, sol, returns x, c & p arrays")
        c_, phi = sol.split()
        xu = mesh.coordinates()
        cu = c_.compute_vertex_values(mesh)
        pu = phi.compute_vertex_values(mesh)
        xyz = np.array(
            [(xu[j], cu[j], pu[j]) for j in range(len(xu))],
            dtype=[("x", "d"), ("c", "d"), ("phi", "d")],
        )
        xyz.sort(order="x")
        return xyz["x"], xyz["c"], xyz["phi"]
    
    def TRBDF2ta(self, **kwargs):
        return 0
    
    def func_space(self, **kwargs):
        return 1

    
    
# %% Final Class for FEM    
class client_code(BaseClass):
    def __init__(self, factory: AbstractSettings, **kwargs):
        """
        The client code works with factories and products only through abstract
        types: AbstractSettings and AbstractProduct. This lets you pass any factory
        or product subclass to the client code without breaking it.
        """
        self.factory = factory
        
        self.ions = kwargs.get("ions", Ion())
        self.conc = kwargs.get("conc", IonConcentration())
        
        self._flux_in = factory.flux_in()
        self._flux_out = factory.flux_out()
        self._form = factory.variational_form()
        
    @property
    def flux_in(self)->float:
        """Return k_B*T given e_r. Units: Coulomb*V"""
        return self._flux_in.val(C_0=self.conc.C_s, C_1=self.conc.C, rate=1e-12, m=1)
    
    @property
    def flux_out(self)->float:
        """Return k_B*T given e_r. Units: Coulomb*V"""
        return self._flux_out.val(C_0=self.conc.C, C_1=self.conc.C_out, rate=1e-12, m=1)
    # @kbT.setter
    # def kbT(self, _): pass 


# %% Operations
if __name__ == "__main__":
    from pathlib import Path
    test_Na1 = Ion(1e-16, 1, 80, CtoK=True)
    # test_Na2 = Ion(1e-16, 1, 80, CtoK=True).calc_screen(2.95)
    
    # 1 -> unafected, 1_1 == 1_2 != 1_3
    # test_Na1_1 = test_Na1.copy()
    # test_Na1_2 = test_Na1_1.update(("z",2), ("D",1e-12))
    # test_Na1_3 = test_Na1_1.copy().update(("z",-1), ("D",1e-14))
    
    fem_kwarg_keys = {
        "diffusivity": "D",
        "tempC": "T",
        "valence": "z",
        "csurf": "C_s",
        "cbulk": "C_b",
        "thick_na": "L_j0",
        "cinit": "C_0",
        "screen": "screen",        
        }
    
    fem_kwargs = {
        "diffusivity": 1e-16,
        "tempC": 80,
        "valence": 1.5,
        "csurf": 4.48e19,
        "cbulk": 1e-20,
        "thick_na": 1e-10,
        "cinit": 1e12,
        "screen": 1e-7,        
        }
    
    # test_Na3 = Ion().sanitize(fem_kwargs, None, True, CtoK=True)
    # test_Na4 = Ion().sanitize(fem_kwargs, fem_kwarg_keys, True, CtoK=True)
    test_Na5_1 = Ion().sanitize(fem_kwargs, fem_kwarg_keys, True, CtoK=True)
    test_Na5_2 = IonConcentration().sanitize(fem_kwargs, fem_kwarg_keys, True)

    layer=client_code(const_flux_w_bias(), ions=test_Na5_1, conc=test_Na5_2)
    
    