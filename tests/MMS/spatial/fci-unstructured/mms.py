#!/usr/bin/env python3
#
# Generate manufactured solutions for the unstructured FCI test
#
# Same manufactured solutions as ../fci/mms.py. On unstructured grids the
# nodes of each poloidal plane are stored along x with nz = 1, so BOUT++'s
# index coordinates x and z are not geometric. The expressions are therefore
# written in terms of xu and zu, which data/BOUT.inp defines from the grid
# coordinates R and Z (input:grid_variables), so that xu and zu equal x and z
# of the structured test.

from math import pi
import warnings

from boututils.boutwarnings import AlwaysWarning
from boutdata.data import BoutOptionsFile
from boutdata.mms import diff, exprToStr, x, y, z
from sympy import Expr, Symbol, cos, sin, sqrt

warnings.simplefilter("ignore", AlwaysWarning)

f = sin(y - z) + sin(y - 2 * z)
K = cos(z - y)

Lx = 0.1
Ly = 10.0
Lz = 1.0

Bt = 1.0
Bp = 0.05
Bpprime = 0.1

Bpx = Bp + (x - 0.5) * Lx * Bpprime  # Note: x in range [0,1]
B = sqrt(Bpx**2 + Bt**2)


def FCI_grad_par(f: Expr) -> Expr:
    return (Bt * diff(f, y) * 2.0 * pi / Ly + Bpx * diff(f, z) * 2.0 * pi / Lz) / B


def FCI_grad2_par2(f: Expr) -> Expr:
    return FCI_grad_par(FCI_grad_par(f))


def FCI_div_par(f: Expr) -> Expr:
    return Bpx * FCI_grad_par(f / Bpx)


def FCI_div_par_K_grad_par(f: Expr, K: Expr) -> Expr:
    return (K * FCI_grad2_par2(f)) + (FCI_div_par(K) * FCI_grad_par(f))


def FCI_Laplace_par(f: Expr) -> Expr:
    return FCI_div_par(FCI_grad_par(f))


def div_a_grad_pol(a: Expr, f: Expr) -> Expr:
    """Poloidal-plane divergence div_pol(a grad_pol f), with R = (x - 1/2) Lx
    and Z = z Lz / (2 pi), as computed by Div_a_Grad_perp_FCI_unstructured
    (rho_s0 = 1)."""
    dR = 1.0 / Lx
    dZ = 2.0 * pi / Lz
    return dR**2 * diff(a * diff(f, x), x) + dZ**2 * diff(a * diff(f, z), z)


############################################
# Equations solved

xu, zu = Symbol("xu"), Symbol("zu")

options = BoutOptionsFile("data/BOUT.inp")

for name, expr in (
    ("input_field", f),
    ("K", K),
    ("grad_par_solution", FCI_grad_par(f)),
    ("grad2_par2_solution", FCI_grad2_par2(f)),
    ("div_par_solution", FCI_div_par(f)),
    ("div_par_K_grad_par_solution", FCI_div_par_K_grad_par(f, K)),
    ("div_par_K_grad_par_mod_solution", FCI_div_par_K_grad_par(f, K)),
    ("laplace_par_solution", FCI_Laplace_par(f)),
    ("FV_div_par_mod_solution", FCI_div_par(f * K)),
    ("FV_div_par_fvv_solution", FCI_div_par(f * K * K)),
    ("FV_div_a_grad_perp_solution", div_a_grad_pol(K, f)),
    ("FV_div_a_grad_perp_upwind_solution", div_a_grad_pol(K, f)),
    # Z component of the flux vector -K grad_pol f returned as flow_zlow
    ("FV_div_a_grad_perp_flowZ_solution", -K * diff(f, z) * 2.0 * pi / Lz),
):
    expr_str = exprToStr(expr.subs({x: xu, z: zu}, simultaneous=True))
    print(f"{name} = {expr_str}")
    options[name] = expr_str

options.write("data/BOUT.inp", overwrite=True)
