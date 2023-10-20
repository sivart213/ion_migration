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
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
# from collections import UserDict
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, _FIELDS, _FIELD, InitVar # astuple, asdict
import scipy.constants as constants

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



# %% Concentration cflux Products
class AbstractCFlux(ABC, BaseClass):
    @abstractmethod
    def val(self):
        pass
    
    def local_bcs(self, **kwargs):
        return None
    
    # TODO add method for parsing 2 layer Conc classes plus rate

class Flux0(AbstractCFlux):
    def val(self, **kwargs):
        return 0

class FluxConst(AbstractCFlux):
    def val(self, C_0, **kwargs):
        return C_0
    
    def local_bcs(self, func_space, C_0, mesh_func, index):
        print(f"return dlf.DirichletBC({func_space}, {C_0}, {mesh_func}, {index})")
    
class FluxRate(AbstractCFlux):
    def val(self, C_0, rate, **kwargs):
        return rate * C_0

class FluxInterface(AbstractCFlux):
    def val(self, C_0, C_1, rate, m):
        return rate * (C_0 - C_1 / m)

# class AbstractVFlux(ABC, BaseClass):
#     def __init__(self, C_0=0, C_1=0, rate=0, m=1):
#         self.C_0 = C_0
#         self.C_1 = C_1
#         self.rate = rate
#         self.m = m
    
#     @property
#     @abstractmethod
#     def val(self):
#         pass

# class VFlux0(AbstractVFlux):
#     @property
#     def val(self):
#         return 0
# %% Voltage Flux Products
class AbstractVFlux(ABC, BaseClass):
    def __init__(self, C_0=0, C_1=0, rate=0, m=1):
        self.C_0 = C_0
        self.C_1 = C_1
        self.rate = rate
        self.m = m
    
    @property
    @abstractmethod
    def val(self):
        pass
    
    # TODO add method for parsing 2 layer Conc classes plus rate

class VFlux0(AbstractVFlux):
    @property
    def val(self):
        return 0

class VFluxConst(AbstractVFlux):
    @property
    def val(self):
        return self.C_0

class VFluxRate(AbstractVFlux):
    @property
    def val(self):
        return self.rate * self.C_0

class VFluxInterface(AbstractVFlux):
    @property
    def val(self):
        return self.rate * (self.C_0 - self.C_1 / self.m)
    

# %% Variational form Products
class AbstractForm(ABC, BaseClass):
    @abstractmethod
    def func(self):
        pass
    
    def get_array(self, mesh, sol):
        return "takes mesh, sol, returns x, c & p arrays"
    
    def TRBDF2ta(self, **kwargs):
        return None


class linear_diffusion(AbstractForm):
    def func(self, a):
        print(f"Takes flux_in&out (w cftrial), mob*cftrial*E_in&out, cftest, dx, and ds")
        print(f"Returns Anp terms 1 & 2")
        return 5*a
    
    def get_array(self, mesh, sol):
        return "takes mesh, sol, returns x & c arrays"

class linear_NP(AbstractForm):
    def func(self, a):
        res = linear_diffusion().func(a)+6/a
        print(f"Additionally takes phiftrial and phiftest")
        print(f"To include Anp terms 3 & 4 & Ap terms 1 & 2")
        return res

class linear_NP_alt(AbstractForm):
    def func(self, a):
        res = linear_diffusion().func(a)+6/a
        print(f"Additionally takes phiftrial")
        print(f"To include Anp terms 3 & 4")
        return res
    # def get_array(self, mesh, sol):
    #     return mesh, sol[0], sol[1]

class linear_PNP(AbstractForm):
    def func(self, a):
        res = linear_NP().func(a)+a**2
        print(f"Additionally takes and screening exp")
        print(f"To include Ap terms 3")
        return res

    # def get_array(self, mesh, sol):
    #     return mesh, sol[0], sol[1]
    
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

#%% build form?
class Builder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """

    @property
    @abstractmethod
    def product(self) -> None:
        pass

    @abstractmethod
    def set_flux_in_as_0(self) -> None:
        pass

    @abstractmethod
    def set_flux_in_as_const(self) -> None:
        pass

    @abstractmethod
    def set_flux_in_as_rate(self) -> None:
        pass
    
    @abstractmethod
    def set_flux_out_as_0(self) -> None:
        pass

    @abstractmethod
    def set_flux_out_as_const(self) -> None:
        pass

    @abstractmethod
    def set_flux_out_as_rate(self) -> None:
        pass


class ConstructSettings(Builder):
    def __init__(self):
        self.reset()

    def reset(self):
        self._product = BaseSettings
    
    @property
    def product(self):
        product = self._product
        self.reset()
        return product

    def set_flux_in_as_0(self):
        class ManualSettings(self._product):
            def flux_in(self):
                return Flux0()
        self._product = ManualSettings

    def set_flux_in_as_const(self):
        class ManualSettings(self._product):
            def flux_in(self):
                return FluxConst()
        self._product = ManualSettings


    def set_flux_in_as_rate(self):
        class ManualSettings(self._product):
            def flux_in(self):
                return FluxRate()
        self._product = ManualSettings

    def set_flux_out_as_0(self):
        class ManualSettings(self._product):
            def flux_out(self):
                return Flux0()
        self._product = ManualSettings

    def set_flux_out_as_const(self):
        class ManualSettings(self._product):
            def flux_out(self):
                return FluxConst()
        self._product = ManualSettings


    def set_flux_out_as_rate(self):
        class ManualSettings(self._product):
            def flux_out(self):
                return FluxRate()
        self._product = ManualSettings

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
    
    builder = ConstructSettings()
    builder.set_flux_in_as_const()
    builder.set_flux_out_as_rate()
    # alt_fact = builder.product
    
    # layer2=client_code(alt_fact(), ions=test_Na5_1, conc=test_Na5_2)
    