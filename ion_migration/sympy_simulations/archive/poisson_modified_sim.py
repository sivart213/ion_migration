# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 18:05:04 2023

@author: j2cle
"""

import sympy as sp
import numpy as np
import pandas as pd
from utilities import Length, Q_E, PERMI__CM


class Poisson_Sim(object):
    def __init__(
        self, x_L_na=0.005, x_L=0.045, er=2.95, z=1, c_surf=1e18, c_base=1e-20, bias=0, gnd=0, n_depth=1000, **kwargs,
    ):
        self.f = sp.symbols('f', cls=sp.Function)
        self.x = sp.Symbol('x[0]', real=True)

        self.x_L_na = x_L_na
        self.x_L = x_L
        self.er = er
        self.z = z
        self.c_surf = c_surf
        self.c_base = c_base
        self.depth = n_depth
        self.bias = bias
        self.gnd = gnd
        self.x_0 = kwargs.get("x_0", 0)

    @property
    def c_surf(self):
        """Return the surface concentration"""
        return self._c_surf

    @c_surf.setter
    def c_surf(self, val):
        """Set the surface concentration"""
        if val is not None:
            self._c_surf = val
            self.piecewise()

    @property
    def x_L_na(self):
        """Return the surface concentration"""
        return self._x_L_na

    @x_L_na.setter
    def x_L_na(self, val):
        """Set the surface concentration"""
        if val is not None:
            self._x_L_na = val
            self.piecewise()

    @property
    def atom_conc(self):
        """Return the piecewise as an atomic concentration"""
        if not hasattr(self, "_atom_conc"):
            self._atom_conc = None
        return self._atom_conc

    @property
    def ion_conc(self):
        """Return the piecewise as a charge concentration"""
        if not hasattr(self, "_ion_conc"):
            self._ion_conc = None
        return self._ion_conc

    # @property
    # def ode_bias(self):
    #     """Return the piecewise as a charge concentration"""
    #     return sp.Eq(self.f(self.x).diff(self.x, self.x), 0)

    @property
    def f_bias(self):
        """Return the piecewise as a charge concentration"""
        ode_bias = sp.Eq(self.f(self.x).diff(self.x, self.x), 0)
        return sp.dsolve(ode_bias,
                            self.f(self.x),
                            simplify=False,
                            ics={self.f(0): self.bias, self.f(self.x_L): self.gnd})

    @property
    def voltage_bias(self):
        """Return the piecewise as a charge concentration"""
        if not hasattr(self, "_voltage_bias"):
            self._voltage_bias = self.f_bias.rhs
        return self._voltage_bias

    @property
    def efield_bias(self):
        """Return the piecewise as a charge concentration"""
        if not hasattr(self, "_efield_bias"):
            self._efield_bias = self.f_bias.rhs.diff(self.x) * -1
        return self._efield_bias

    @property
    def depth(self):
        """Return the surface concentration"""
        if not hasattr(self, "_depth"):
            self._depth = np.linspace(0, self.x_L, 1000)
        return self._depth

    @depth.setter
    def depth(self, val):
        """Set the surface concentration"""
        if val is not None:
            self._depth = np.linspace(self.x_L * -0.01, self.x_L, val)


    def piecewise(self):
        try:
            i_conc = sp.Piecewise((0, self.x < 0),
                                  (self.c_surf * Q_E * self.z/(self.er * PERMI__CM), self.x <= self.x_L_na),
                                  (self.c_base * Q_E * self.z/(self.er * PERMI__CM), self.x <= self.x_L),
                                  (0, self.x > self.x_L),
                                  evaluate=False)
            self._ion_conc = i_conc

            a_conc = sp.Piecewise((0, self.x < 0),
                                  (self.c_surf, self.x <= self.x_L_na),
                                  (self.c_base, self.x <= self.x_L),
                                  (0, self.x > self.x_L),
                                  evaluate=False)
            self._atom_conc = a_conc

        except AttributeError:
            return
        return


    def solve(self, get_field=True, boundaries=None):

        if self.atom_conc is None or self.atom_conc is None:
            self.piecewise()

        ode_v = sp.Eq(self.f(self.x).diff(self.x,self.x), -1*self.ion_conc)
        eion_s = float(sp.integrate(self.x/self.x_L*ode_v.rhs, (self.x, 0, self.x_L)))

        if boundaries is None:
            d_bound = {self.f(self.x_L) : self.gnd,
                       self.f(self.x).diff(self.x).subs(self.x, self.x_L) : float(-1*self.efield_bias + eion_s)}
        else:
            d_bound = {}
            for bound in boundaries:
                if bound[0] == 0:
                    d_bound[self.f(bound[1])] = self.voltage_bias.subs(self.x, bound[1]) + bound[2]
                elif bound[0] == 1:
                    d_bound[self.f(self.x).diff(self.x).subs(self.x, bound[1])] = float(-1*self.efield_bias + bound[2])

        self.res = sp.dsolve(ode_v, self.f(self.x), simplify=False, ics=d_bound)

        self.voltage = self.res.rhs
        self.efield = self.res.rhs.diff(self.x) * -1

    def array(self, parameter):

        if parameter.lower() in ["atom_conc", "ion_conc", "voltage", "efield", "voltage_bias", "efield_bias"]:
            param = getattr(self, parameter.lower())
            return sp.lambdify(self.x, param)(self.depth)

test = Poisson_Sim(Length(80, "nm").cm, x_L=Length(80, "um").cm, c_surf=1e16)


test.solve()
var_v_bias = test.array("voltage_bias")
var_e_bias = test.array("efield_bias")
var_v = test.array("voltage")
var_e = test.array("efield")

# pd.DataFrame(dict(x=test.depth*1e4, y=var_e), columns=["x","y"]).plot(x="x",y="y", grid=True)
# pd.DataFrame(dict(x=test.depth*1e4, y=(var_e-var_e.min())/var_e_bias*100), columns=["x","y"]).plot(x="x",y="y", grid=True)
# pd.DataFrame(dict(x=test.depth*1e4, y=(var_e-var_e_bias)/var_e_bias*100), columns=["x","y"]).plot(x="x",y="y", grid=True)

# pd.DataFrame(dict(x=test.depth*1e4, y=var_v), columns=["x","y"]).plot(x="x",y="y", grid=True)
# pd.DataFrame(dict(x=test.depth*1e4, y=var_v_bias), columns=["x","y"]).plot(x="x",y="y", grid=True)
# pd.DataFrame(dict(x=test.depth*1e4, y=(var_v-var_v_bias)/var_v_bias*100), columns=["x","y"]).plot(x="x",y="y", grid=True)

# xn = [(50, "um"), (450, "um"), 0.005]
# xnn = [Length(i[0], i[1]).cm if isinstance(i, (list, tuple))  else i for i in xn]
