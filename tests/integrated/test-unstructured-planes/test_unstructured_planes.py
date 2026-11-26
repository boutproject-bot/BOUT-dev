#!/usr/bin/env python3
#
# The unstructured perpendicular operator with a structured z (nz > 1): the
# stored matrices act within a plane of cells (the cell numbers count one
# z plane), so BOUT++ applies them to each z plane in turn.
#
# A zoidberg slab (one plane of cells with Dirichlet walls in R) is run
# with nz = 1 and, copied with every 3D variable repeated in z, with
# nz = 4, on 1 and 2 processors. On plane k the test program scales f by
# (1 + k) and a by (1 + k / 2). Div(a Grad_perp f), with and without
# upwinding, its R and Z flows and the flow into the walls on plane k must
# be (1 + k)(1 + k / 2) times the nz = 1 results. FCI, whose maps connect
# the cells of one plane, must refuse nz > 1 on unstructured meshes.

import pathlib
import shutil

import numpy as np
from boutdata.collect import collect
from boututils.run_wrapper import build_and_log, launch, launch_safe
from netCDF4 import Dataset
from zoidberg.field import Slab
from zoidberg.unstructured import UnstructuredPoloidalGrid, make_unstructured_grid

NZ = 4
TOLERANCE = 1e-12
OUTPUTS = ("result", "result_upwind", "flow_R", "flow_Z", "wall")


def copy_grid(source, target, nz, transform="identity"):
    """The grid with nz planes (3D variables repeated in z), labelled for
    the identity transform by default: the operator does not need parallel
    slices"""
    with Dataset(source) as src, Dataset(target, "w") as dst:
        for name, dim in src.dimensions.items():
            dst.createDimension(name, nz if name == "z" else len(dim))
        for attribute in src.ncattrs():
            dst.setncattr(attribute, src.getncattr(attribute))
        dst.setncattr("parallel_transform", transform)
        for name, var in src.variables.items():
            values = var[...]
            if "z" in var.dimensions:
                axis = var.dimensions.index("z")
                assert values.shape[axis] == 1, f"{name} has nz > 1"
                values = np.repeat(values, nz, axis=axis)
            if name == "nz":
                values = nz
            dst.createVariable(name, var.dtype, var.dimensions)[...] = values


def run(name, nproc, args):
    path = pathlib.Path(name)
    path.mkdir(exist_ok=True)
    shutil.copy("data/BOUT.inp", path)
    status, out = launch_safe(
        f"./test_unstructured_planes -d {name} {args}", nproc=nproc, pipe=True
    )
    path.joinpath("run.log").write_text(out)
    assert status == 0, out
    return {
        var: collect(var, path=name, info=False, xguards=False, yguards=False)
        for var in OUTPUTS
    }


def test_unstructured_planes():
    pol = UnstructuredPoloidalGrid.slab(
        5, 4, 1.0, 1.0, R0=1.5, jitter=0.2, jitter_dirs="z", periodic_Z=True, seed=7
    )
    field = Slab(By=1.0, Bz=0.0, Bzprime=0.0)
    make_unstructured_grid("plane.nc", pol, field, ny=2, Ly=10.0, dirichlet_walls=True)
    copy_grid("plane.nc", "plane_nz1.nc", 1)
    copy_grid("plane.nc", f"plane_nz{NZ}.nc", NZ)
    build_and_log("unstructured operator on z planes")

    one = run("nz1", 1, "mesh:file=plane_nz1.nc")
    failures = []
    for var in OUTPUTS:
        size = np.max(np.abs(one[var]))
        print(f"nz = 1: max |{var}| = {size:.3f}")
        if size < 0.01:
            failures.append(f"{var} is too small to test ({size:.2e})")
    scale = np.array([(1 + k) * (1 + k / 2) for k in range(NZ)])
    for nproc in (1, 2):
        planes = run(
            f"nz{NZ}_np{nproc}", nproc, f"NYPE={nproc} mesh:file=plane_nz{NZ}.nc"
        )
        for var in OUTPUTS:
            expected = one[var][..., :1] * scale
            error = np.max(np.abs(planes[var] - expected)) / np.max(np.abs(one[var]))
            print(
                f"nz = {NZ}, {nproc} rank(s): {var} differs by {error:.2e} (relative)"
            )
            if error > TOLERANCE:
                failures.append(f"{nproc} rank(s): {var} differs by {error:.2e}")

    # FCI's maps connect the cells of one plane: with nz > 1 it must stop
    copy_grid("plane.nc", f"plane_fci_nz{NZ}.nc", NZ, transform="fci")
    path = pathlib.Path(f"fci_nz{NZ}")
    path.mkdir(exist_ok=True)
    shutil.copy("data/BOUT.inp", path)
    status, out = launch(
        f"./test_unstructured_planes -d {path} mesh:file=plane_fci_nz{NZ}.nc "
        "mesh:paralleltransform:type=fci "
        "mesh:paralleltransform:xzinterpolation:type=precomputed",
        nproc=1,
        pipe=True,
    )
    path.joinpath("run.log").write_text(out)
    if status == 0 or "FCI on unstructured meshes needs nz = 1" not in out:
        failures.append(f"FCI with nz = {NZ} did not stop with the expected error")
    assert not failures, failures
