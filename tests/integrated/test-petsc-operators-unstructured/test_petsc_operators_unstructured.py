# Support operators on an unstructured grid with parallel walls:
#
# * the cell volume is dy * dagp_us_volume (not J dy dagp_us_volume);
# * for f linear along a straight field (d f / d l = 1) and K = 1, the
#   divergence Div_par(K Grad_par f) vanishes everywhere with
#   bndry_flux = true (the full boundary flux enters and leaves), and with
#   bndry_flux = false it is -1/dl and +1/dl at the cells next to the upper
#   and lower walls (the flux through the walls is removed);
# * the assembled Div_par_Grad_par equals the bndry_flux = true result.
#
# Run on 1 rank and on 2 ranks (split in y, so that the walls are on
# different ranks).

import pathlib
import shutil

import numpy as np
from boutdata.collect import collect
from boututils.run_wrapper import build_and_log, launch_safe
from zoidberg.field import Slab
from zoidberg.unstructured import UnstructuredPoloidalGrid, make_unstructured_grid


def run(nproc, args):
    path = f"data_np{nproc}"
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
    failures = []
    for name, value in checks.items():
        ok = value < 1e-10
        print(f"{path}: {name}: {value:.2e}\t... {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)
    return failures


def test_petsc_operators_unstructured():
    NY = 16
    pol = UnstructuredPoloidalGrid.slab(
        5, 4, 1.0, 1.0, R0=1.5, jitter=0.2, jitter_dirs="z", periodic_Z=True, seed=7
    )
    make_unstructured_grid(
        "grid.fci.nc",
        pol,
        Slab(By=1.0, Bz=0.0, Bzprime=0.0),
        ny=NY,
        Ly=10.0,
        walls_y=True,
    )
    build_and_log("Support operators on an unstructured grid")

    failures = []
    for nproc, args in ((1, "NYPE=1"), (2, "NYPE=2")):
        failures += [f"{nproc} rank(s): {f}" for f in check(run(nproc, args))]
    assert not failures, failures
