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
import re
import sympy as sp
import numpy as np
import pandas as pd
import sympy.physics.units as su

from research_tools.equations import screened_permitivity, debye_length
from research_tools.functions import get_const, map_plt, convert_val, get_const, all_symbols, has_units

# %% Code
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

def charge_concentration(C, z):
    if isinstance(C, (tuple,list)):
        return [charge_concentration(var, z) for var in C]
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    q = get_const("elementary_charge", *([True] if all_symbols(arg_in) else [w_units, ["C"]]))
    return q*z*C

def poisson_rhs(C, z, epsilon_r):
    if isinstance(C, (tuple,list)):
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

def poisson_ode2(C, z, epsilon_r, L=None, f=None, x=None):
    arg_in = vars().copy()

    if f is None:
        f = sp.symbols("f", cls=sp.Function)
    if x is None:
        x = sp.Symbol("x", real=True)

    w_units = has_units(arg_in)

    ion = poisson_rhs(C, z, epsilon_r)
    if L is not None:
        # ion = sp.Piecewise(sp.ITE(sp.And(x>=0, x<=L),ion,0), evaluate=False)
        ion = list(ion) if isinstance(ion, (tuple, list)) else [ion]
        if not isinstance(L, (tuple, list)):
            args = [(ion[0], (0, L))]
        elif (
            any([isinstance(n, bool) for n in L])
            or (len(ion)==1 and len(L)<=2 and not isinstance(L[0], (tuple, list, str)))
            ):
            args = [(ion[0], L)]

        else:
            args = [(ion[n], L[n]) for n in range(len(ion))]
            args.append((0,) + tuple(L[n] for n in range(len(ion), len(L))))
            # if len(ion) < len(L):
            #     args.append((0,) + tuple(L[n] for n in range(len(ion), len(L))))

        ion = make_piecewise(*args, var=x)

    return sp.Eq(f(x).diff(x,x), -1*ion)

def poisson_integral(z, C, er, L, l_D=None, **kwargs):
    x = sp.Symbol("x", real=True)

    # Constants
    e0 = get_const("e0", False, [su.farad, su.cm])

    if l_D is None:
        l_D = debye_length(z, er, C, kwargs.get("T", 300))

    if l_D == 0:
        orig_ion = C * e0 * z / er
    else:
        orig_ion = C * e0 * z / er * sp.exp(x * l_D)

    res = sp.integrate(-1*sp.integrate(orig_ion, (x, 0, L)), (x, 0, L))

    return res


def extract_variable(expr, targ):
    if "sympy" not in str(type(expr)):
        return expr
    res = [var for var in expr.free_symbols if targ == str(var)]
    if len(res) == 1:
        return res[0]
    for ar in expr.args:
        if targ == str(ar.func):
            return ar.func
        if "function" in str(ar.func) or "relational" in str(ar.func):
            res.append(extract_variable(ar, targ))
    if len(res) == 0:
        return [var for var in expr.free_symbols if targ in str(var)]
    if len(res) == 1:
        return res[0]
    return res


# %% Operations
if __name__ == "__main__":
    from research_tools.functions import p_find, save

    f = sp.symbols("f", cls=sp.Function)
    x = sp.Symbol("x", real=True)

    # includes 0
    C, C_B, V, l_D = sp.symbols("C_Na C_B V k_0", real=True, constant=True, nonnegative=True)
    # does not include 0
    er, z, T = sp.symbols("epsilon_r z T", real=True, constant=True, positive=True)
    L, L1, L2, Ln = sp.symbols("L L_Na L_B n", real=True, constant=True, positive=True)

    # c1, c2, c3 = sp.symbols("C1 C2 C3", real=True, constant=True)

    thick_na = convert_val(0.5, "um", "cm")
    thick_eva = convert_val(450, "um", "cm")
    thick_ratio = thick_na / thick_eva

    depth = np.linspace(0, thick_eva/100, 250)
    depth = np.concatenate((depth, np.linspace(max(depth), thick_eva, 3*len(depth)+1)[1:]))

    base = {
        su.e0: get_const("e0", False, ["F", "cm"]),
        su.q: get_const("elementary_charge", False, ["C"]),
        V: 1500,
        z: 1,
        er: 2.95,
        C: 2.24e19,
        C_B: 1e-20,
        L: thick_eva,
        L1: thick_na,
        L2: thick_eva-thick_na,
        Ln: thick_ratio,
        }
    
    
    Conc = make_piecewise((C, 0, L1), (0, L1, L1+L2, True), 0)
    res = sp.integrate(sp.integrate(-1*poisson_rhs(Conc, z, er) , (x)), (x, 0, L1+L2))
    # sp.piecewise_exclusive((make_piecewise((C, 0, L/2, True, False), (C_B, L/2, L), 0)*er).simplify())

    # eqs = {
    #     # 0: (make_piecewise((-C/(2*L1)*x+C, 0, L1), 0), L1),
    #     # 1: (make_piecewise((C, 0, L1), 0), L1),
    #     # 2: (make_piecewise((C/2, L1, 2*L1), 0), 2*L1),
    #     # 3: (make_piecewise((-C/(2*L1)*x+C, 0, L1), (C/2, L1, 2*L1, True), 0), 2*L1),
    #     # 4: (make_piecewise((-C/(2*L1)*x+C, 0, L1), (C/2, L1, 2*L1, True), 0), 4*L1),
    #     # 5: (make_piecewise((C, 0, L1), (C/2, L1, 2*L1, True), (0, 2*L1, L, True), 0), L),

    #     # 0: (C, L1),
    #     # 1: (C, Ln*L),
    #     # 2: (C, thick_na),

    #     0: (make_piecewise((C, 0, L1), 0), L1),
    #     1: (make_piecewise((C, 0, Ln*L), 0), Ln*L),
    #     2: (make_piecewise((C, -L1/2, L1/2), 0), L1),

    #     3: (make_piecewise((C, 0, L1), 0), L1+L2),
    #     4: (make_piecewise((C, 0, Ln*L), 0), L),
    #     5: (make_piecewise((C, 0, thick_na), 0), thick_eva),

    #     6: (make_piecewise((C, 0, L1), (0, L1, L, True), 0), L),
    #     7: (make_piecewise((C, 0, Ln*L), (0, Ln*L, L, True), 0), L),

    #     # 8: (make_piecewise((C, 0, L1), (C_B, L1, L1+L2, True), 0), L1+L2),
    #     # 9: (make_piecewise((C, 0, Ln*L), (C_B, Ln*L, L, True), 0), Ln*L),
    #     }

    # res = {}
    # res_dfs = {}
    # n = 0
    # # for eq in eqs:
    # for n in range(len(eqs)):
    #     print(n)

    #     # res[f"run_{n}"]

    #     r = res[n] = {}

    #     r["ode"] = poisson_ode(sp.piecewise_exclusive(eqs[n][0]), z, er, f, x)

    #     r["ion"] = (r["ode"].rhs * -1).simplify()

    #     r["qtot"] = sp.integrate(r["ion"], (x, 0, eqs[n][1]))
    #     r["qtot_alt"] = sp.integrate(r["ion"], (x))
    #     r["qbar"] = sp.integrate(x*r["ion"], (x, 0, eqs[n][1]))
    #     r["qbar_alt"] = sp.integrate(x*r["ion"], (x))

    #     r["xbar"] = r["qbar"]/r["qtot"]
    #     r["xbar_alt"] = r["qbar_alt"] /r["qtot_alt"]

    #     r["qgx"] = (1-r["xbar"]/eqs[n][1])*r["qtot"]
    #     r["qsx"] = (r["xbar"]/eqs[n][1])*r["qtot"]

    #     # r["qg"] = sp.integrate(r["ion"], (x, 0, r["xbar"]))
    #     # r["qs"] = -1 * sp.integrate(r["ion"], (x, r["xbar"], eqs[n][1]))

    #     r["qs"] = -1 * sp.integrate((x)/(eqs[n][1]) * r["ion"], (x, 0, eqs[n][1]))
    #     r["qg"] = sp.integrate((x-eqs[n][1])/(eqs[n][1]) * r["ion"], (x, 0, eqs[n][1]))

    #     r["qs_init"] = -1 * sp.integrate((x)/(eqs[n][1]) * r["ion"], (x, 0, eqs[n][1]))
    #     r["qg_init"] = sp.integrate((x-eqs[n][1])/(eqs[n][1]) * r["ion"], (x, 0, eqs[n][1]))


    #     r["qs_alt"] = -1 * sp.integrate((x)/(eqs[n][1]) * r["ion"], (x))
    #     r["qg_alt"] = sp.integrate((x-eqs[n][1])/(eqs[n][1]) * r["ion"], (x))

    #     # r["sol_E"] = sp.dsolve(sp.piecewise_exclusive(poisson_ode(eqs[n][0], z, er, f, x, 1)), f(x), simplify=False).rhs  # Error in this when numbers go in
    #     # r["sol_Cg"] = sp.dsolve(r["ode"], f(x), simplify=False, ics={f(eqs[n][1]) : 0, f(x).diff(x).subs(x, eqs[n][1]) : -1*(r["qtot"]/2)}).rhs.simplify()
    #     # r["solg"] = sp.dsolve(r["ode"], f(x), simplify=False, ics={f(eqs[n][1]) : 0, f(x).diff(x).subs(x, eqs[n][1]) : -1*(V/eqs[n][1] + r["qtot"]/2)}).rhs.simplify()

    #     # r["sol_V"] = sp.dsolve(sp.Eq(f(x).diff(x,x), 0), f(x), simplify=False, ics={f(eqs[n][1]) : 0, f(x).diff(x).subs(x, eqs[n][1]) : -1*(V/eqs[n][1])}).rhs.simplify()
    #     # r["sol_C"] = sp.dsolve(r["ode"], f(x), simplify=False, ics={f(eqs[n][1]) : 0, f(x).diff(x).subs(x, eqs[n][1]) : -1*(-r["qs"])}).rhs.simplify()
    #     # r["sol"] = sp.dsolve(r["ode"], f(x), simplify=False, ics={f(eqs[n][1]) : 0, f(x).diff(x).subs(x, eqs[n][1]) : -1*(V/eqs[n][1] - r["qs"])}).rhs.simplify()

    #     r["sol_V"] = sp.dsolve(sp.Eq(f(x).diff(x,x), 0), f(x), simplify=False, ics={f(0) : V, f(eqs[n][1]) : 0}).rhs.simplify()
    #     r["sol_C"] = sp.dsolve(r["ode"], f(x), simplify=False, ics={f(0) : 0, f(eqs[n][1]) : 0}).rhs.simplify()
    #     r["sol"] = sp.dsolve(r["ode"], f(x), simplify=False, ics={f(0) : V, f(eqs[n][1]) : 0}).rhs.simplify()

    #     data = [
    #         depth*convert_val(1, "cm", "um"),
    #         sp.lambdify(x, eqs[n][0].subs(base))(depth),
    #         sp.lambdify(x, r["ion"].subs(base))(depth),
    #         sp.lambdify(x, r["sol_C"].diff(x).subs(base))(depth),
    #         sp.lambdify(x, r["sol_V"].diff(x).subs(base))(depth),
    #         sp.lambdify(x, r["sol"].diff(x).subs(base))(depth),
    #         sp.lambdify(x, r["sol_C"].subs(base))(depth),
    #         sp.lambdify(x, r["sol_V"].subs(base))(depth),
    #         sp.lambdify(x, r["sol"].subs(base))(depth),
    #         ]

    #     data = [d if getattr(d, "size", 1) == len(depth) else np.full_like(depth, d) for d in data]

    #     res_dfs[f"res_{n}"]  = pd.DataFrame(np.array(data).T, columns=["x", "C", "zqC/e", "E_ion", "E_bias", "E", "V_ion", "V_bias", "V"])

    # save_pth = p_find(
    #     "Dropbox (ASU)",
    #     "Work Docs",
    #     "Data",
    #     "Analysis",
    #     "Simulations",
    #     "PNP",
    #     "EVA",
    #     base="home",
    # )

    # save(res_dfs,save_pth/"Sympy_res","Poisson_res2")
