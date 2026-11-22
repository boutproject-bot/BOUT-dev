# Support operators on an unstructured grid with parallel walls:
#
# * the cell volume is dy * dagp_us_volume (not J dy dagp_us_volume);
# * for f linear along a straight field (d f / d l = 1) and K = 1, the
#   divergence Div_par(K Grad_par f) vanishes everywhere with
#   bndry_flux = true (the full boundary flux enters and leaves), and with
#   bndry_flux = false it is -1/dl and +1/dl at the cells next to the upper
#   and lower walls (the flux through the walls is removed);
# * the assembled Div_par_Grad_par equals the bndry_flux = true result;
# * with a variable coefficient and bndry_flux = false, the integral of the
#   divergence in the physical volume dy * dagp_us_volume vanishes;
# * the boundary values give the same result when set through the
#   label-based Field3D::applyParallelBoundary (the labels FCI_forward and
#   FCI_backward each name two regions) and in expression form (evaluated at
#   the real boundary position on unstructured meshes).
#
# Run on 1 rank and on 2 ranks (split in y, so that the walls are on
# different ranks).
#
# K2 divergence (Div_par_K_Grad_par_k2, conduction_method = PetscOperatorK2;
# matrices k2_* written into the grid file by zoidberg):
#
# * on the straight-field grid the same checks as for the support operator,
#   plus Div_par_Grad_par_k2 and the Div_par_K_Grad_par_mod dispatch (also
#   with variable K and bndry_flux = false);
# * on a tilted-field grid (constant B_z: field lines cross the irregular
#   plane meshes, so the overlaps are not trivial), f linear along y, i.e.
#   along the field lines, gives an exactly zero divergence with
#   bndry_flux = true and -+ (dy/dl)/dl at the upper/lower walls with
#   bndry_flux = false (the support-operator divergence is not exact here);
# * conservation: with bndry_flux = false the dV integral vanishes, and with
#   bndry_flux = true it equals the boundary flux, which enters through the
#   boundary legs as in the support operator (same integral);
# * on a grid without the K2 matrices, both entry points throw an error that
#   names the missing matrices;
# * normalised metric (length_normalisation = L: g_ij / L^2, J / L^3, as
#   Hermes-3 does on FCI grids): the K2 weights in the grid file contain
#   1/dl of the unnormalised metric and must be converted to the mesh
#   lengths. The K2 checks hold with the normalised dl, and with variable K
#   and the Dirichlet wall values of f the K2 and support-operator results
#   are those without normalisation times L^2.
#
# If the generated grids have no K2 matrices (zoidberg without K2 support),
# the K2 checks on them are skipped with a message, and the test still
# passes; set BOUT_TEST_REQUIRE_K2=1 to make that a failure. For development,
# BOUT_TEST_K2_HOOK names a command that adds the matrices to a grid file,
# called as `$BOUT_TEST_K2_HOOK FILE FAMILY` with FAMILY "petsc_slab" or
# "petsc_slab_tilted".

import os
import pathlib
import shlex
import shutil
import subprocess

import numpy as np
from boutdata.collect import collect
from boututils.run_wrapper import build_and_log, launch_safe
from netCDF4 import Dataset
from zoidberg.field import Slab
from zoidberg.unstructured import UnstructuredPoloidalGrid, make_unstructured_grid


def run(nproc, args, path=None):
    path = path or f"data_np{nproc}"
    pathlib.Path(path).mkdir(exist_ok=True)
    shutil.copy("data/BOUT.inp", path)
    status, out = launch_safe(
        f"./test_petsc_operators_unstructured -d {path} {args}", nproc=nproc, pipe=True
    )
    pathlib.Path(path, "run.log").write_text(out)
    assert status == 0, out
    return path


def check(path):
    def get(name):
        return collect(name, path=path, info=False, xguards=False, yguards=False)[
            ..., 0
        ]

    dV, dyV, dl = get("dV"), get("dy_volume"), get("dl")
    div_t, div_f, lap = get("div_bndry"), get("div_nobndry"), get("laplace_par")
    scale = 1.0 / dl.min()

    checks = {
        "dV = dy * dagp_us_volume": np.max(np.abs(dV - dyV)) / np.max(dyV),
        "bndry_flux = true: divergence zero": np.max(np.abs(div_t)) / scale,
        "Div_par_Grad_par = bndry_flux = true result": np.max(np.abs(lap - div_t))
        / scale,
        "bndry_flux = false: interior zero": np.max(np.abs(div_f[:, 1:-1])) / scale,
        "bndry_flux = false: -1/dl at the upper wall": np.max(
            np.abs(div_f[:, -1] + 1.0 / dl[:, -1])
        )
        / scale,
        "bndry_flux = false: +1/dl at the lower wall": np.max(
            np.abs(div_f[:, 0] - 1.0 / dl[:, 0])
        )
        / scale,
    }
    checks["boundary values by label (non-unique labels): divergence zero"] = (
        np.max(np.abs(get("div_bndry_label"))) / scale
    )
    checks["expression boundary values at the boundary point: divergence zero"] = (
        np.max(np.abs(get("div_bndry_expr"))) / scale
    )
    checks["grid variables in boundary expressions: points in the wrong plane"] = float(
        collect("gridvar_mismatch", path=path, info=False)
    )
    div_v = get("div_variable")
    checks["variable K: physical volume integral (bndry_flux = false)"] = abs(
        np.sum(dyV * div_v)
    ) / np.sum(np.abs(dyV * div_v))

    failures = []
    for name, value in checks.items():
        ok = value < 1e-10
        print(f"{path}: {name}: {value:.2e}\t... {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)
    return failures


def report(path, checks, tolerance=1e-10):
    failures = []
    for name, value in checks.items():
        ok = value < tolerance
        print(f"{path}: {name}: {value:.2e}\t... {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)
    return failures


def check_k2(path, tilted):
    """Checks of the K2 divergence (see the top of this file)"""

    def get(name):
        return collect(name, path=path, info=False, xguards=False, yguards=False)[
            ..., 0
        ]

    dyV, dl = get("dy_volume"), get("dl")
    scale = 1.0 / dl.min()
    div_t, div_f = get("k2_div_bndry"), get("k2_div_nobndry")
    # f changes by dy per plane: d f / d l = dy / dl along the field lines
    slope = (10.0 / 16) / dl
    checks = {
        "K2 bndry_flux = true: divergence zero": np.max(np.abs(div_t)) / scale,
        "K2 Div_par_Grad_par_k2 = bndry_flux = true result": np.max(
            np.abs(get("k2_laplace_par") - div_t)
        )
        / scale,
        "K2 Div_par_K_Grad_par_mod(PetscOperatorK2) = Div_par_K_Grad_par_k2": np.max(
            np.abs(get("k2_div_mod") - div_t)
        )
        / scale,
        "K2 bndry_flux = false: interior zero": np.max(np.abs(div_f[:, 1:-1])) / scale,
        "K2 bndry_flux = false: -slope/dl at the upper wall": np.max(
            np.abs(div_f[:, -1] + slope[:, -1] / dl[:, -1])
        )
        / scale,
        "K2 bndry_flux = false: +slope/dl at the lower wall": np.max(
            np.abs(div_f[:, 0] - slope[:, 0] / dl[:, 0])
        )
        / scale,
    }
    div_v = get("k2_div_variable")
    checks[
        "K2 Div_par_K_Grad_par_mod = Div_par_K_Grad_par_k2 (variable K, "
        "bndry_flux = false)"
    ] = np.max(np.abs(get("k2_div_mod_variable") - div_v)) / np.max(np.abs(div_v))
    checks["K2 variable K: physical volume integral (bndry_flux = false)"] = abs(
        np.sum(dyV * div_v)
    ) / np.sum(np.abs(dyV * div_v))
    k2_b, so_b = get("k2_div_variable_bndry"), get("so_div_variable_bndry")
    checks["K2 variable K: volume integral = boundary flux (bndry_flux = true)"] = abs(
        np.sum(dyV * k2_b) - np.sum(dyV * so_b)
    ) / np.sum(np.abs(dyV * k2_b))
    if tilted:
        so = np.max(np.abs(get("div_bndry"))) / scale
        print(f"{path}: (support operator on the tilted grid, not exact: {so:.2e})")
    return report(path, checks)


LENGTH_NORMALISATION = 1.25e-3


def check_normalised(grid, reference):
    """Normalised metric (lengths / L): the K2 checks, and the scaling of
    the variable-K results with Dirichlet wall values (L^2) against the
    unnormalised run in `reference`"""
    L = LENGTH_NORMALISATION
    path = run(
        2,
        f"NYPE=2 mesh:file={grid} length_normalisation={L}",
        f"data_normalised_{pathlib.Path(grid).name.split('.')[0]}",
    )
    failures = check_k2(path, False)

    def get(name, p):
        return collect(name, path=p, info=False, xguards=False, yguards=False)[..., 0]

    checks = {}
    for name in ("k2_div_variable_bndry", "so_div_variable_bndry", "k2_div_variable"):
        ref = get(name, reference)
        checks[f"normalised metric: {name} = unnormalised * L^2"] = np.max(
            np.abs(get(name, path) - ref * L**2)
        ) / np.max(np.abs(ref * L**2))
    return failures + report(path, checks)


def has_k2(filename):
    with Dataset(filename) as ds:
        return "k2_forward_weights" in ds.variables


def make_grid(filename, Bz, family):
    pol = UnstructuredPoloidalGrid.slab(
        5, 4, 1.0, 1.0, R0=1.5, jitter=0.2, jitter_dirs="z", periodic_Z=True, seed=7
    )
    make_unstructured_grid(
        filename,
        pol,
        Slab(By=1.0, Bz=Bz, Bzprime=0.0),
        ny=16,
        Ly=10.0,
        walls_y=True,
    )
    hook = os.environ.get("BOUT_TEST_K2_HOOK")
    if hook and not has_k2(filename):
        subprocess.run(shlex.split(hook) + [filename, family], check=True)
    return has_k2(filename)


def k2_unavailable(filename):
    message = (
        f"SKIPPED: K2 checks on {filename}: the grid has no K2 matrices "
        "(k2_forward_weights); zoidberg does not write them"
    )
    print(message)
    if os.environ.get("BOUT_TEST_REQUIRE_K2", "0") not in ("", "0"):
        return [message + " (BOUT_TEST_REQUIRE_K2 is set)"]
    return []


def check_missing_k2(grid):
    """A grid without the K2 matrices: both entry points throw"""
    nok2 = "grid_nok2.fci.nc"
    shutil.copy(grid, nok2)
    with Dataset(nok2, "a") as ds:
        for name in list(ds.variables):
            if name.startswith("k2_"):
                ds.renameVariable(name, f"disabled_{name}")
    path = run(1, f"NYPE=1 mesh:file={nok2}", path="data_nok2")
    errors = collect("k2_missing_errors", path=path, info=False)
    return report(path, {"K2 missing matrices: clear errors (2 expected)": 2 - errors})


def test_petsc_operators_unstructured():
    k2_straight = make_grid("grid.fci.nc", 0.0, "petsc_slab")
    k2_tilted = make_grid("grid_tilted.fci.nc", 0.3, "petsc_slab_tilted")
    build_and_log("Support operators on an unstructured grid")

    failures = []
    for nproc, args in ((1, "NYPE=1"), (2, "NYPE=2")):
        path = run(nproc, args)
        failures += [f"{nproc} rank(s): {f}" for f in check(path)]
        if k2_straight:
            failures += [f"{nproc} rank(s): {f}" for f in check_k2(path, False)]
    if not k2_straight:
        failures += k2_unavailable("grid.fci.nc")
    if k2_tilted:
        for nproc, args in ((1, "NYPE=1"), (2, "NYPE=2")):
            path = run(
                nproc, f"{args} mesh:file=grid_tilted.fci.nc", f"data_tilted_np{nproc}"
            )
            failures += [f"tilted {nproc} rank(s): {f}" for f in check_k2(path, True)]
    else:
        failures += k2_unavailable("grid_tilted.fci.nc")
    failures += check_missing_k2("grid.fci.nc")
    for grid, ok, ref in (
        ("grid.fci.nc", k2_straight, "data_np2"),
        ("grid_tilted.fci.nc", k2_tilted, "data_tilted_np2"),
    ):
        if ok:
            failures += [f"normalised {grid}: {f}" for f in check_normalised(grid, ref)]
    assert not failures, failures
