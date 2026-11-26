#!/usr/bin/env python3
#
# ShiftedMetricMap (paralleltransform type = shiftedmap): the neighbour of
# each cell along the magnetic field comes from a map in the grid file and
# the toroidal shift to it is applied with FFTs in z.
#
# A sheared slab with closed field lines (4 x 8 cells, nz = 16, zShift
# varying in x and y) is run with ShiftedMetric, and the same cells are
# stored in a shuffled order along x (y of size 1) with the map and run
# with ShiftedMetricMap, on one and on two processors. For every cell the
# forward and backward parallel slices and Grad_par must agree to round-off,
# and the slices must match the field evaluated at the shifted points; the
# shift is not negligible (the slices differ from the unshifted neighbour
# values by order one).

import pathlib
import shutil

import numpy as np
from boutdata.collect import collect
from boututils.run_wrapper import build_and_log, launch_safe
from netCDF4 import Dataset

NX, NY, NZ = 4, 8, 16
TOLERANCE = 1e-12
DY = 2 * np.pi / NY


def field(x, y, z):
    """Band-limited in z (modes 2 and 5 of 8), so that FFT shifts are exact"""
    return np.sin(2 * z + 0.9 * y + x) + 0.4 * np.cos(5 * z - 2 * y + 0.3 * x)


def zshift(x, y):
    """Sheared: the shift depends on x and on y"""
    return 0.7 * (x + 1) * np.sin(y)


X, Y = np.meshgrid(np.arange(NX), np.arange(NY) * DY, indexing="ij")
Z = np.arange(NZ) * 2 * np.pi / NZ
F = field(X[..., None], Y[..., None], Z[None, None, :])


def write(name, nx, ny, values):
    with Dataset(name, "w") as d:
        d.createDimension("x", nx)
        d.createDimension("y", ny)
        d.createDimension("z", NZ)
        for var, value in (("nx", nx), ("ny", ny), ("nz", NZ)):
            d.createVariable(var, "i4")[...] = value
        # Closed field lines: zShift is periodic in y, so no twist at the cut
        d.createVariable("ShiftAngle", "f8", ("x",))[...] = np.zeros(nx)
        for var, value in values.items():
            dims = ("x", "y", "z")[: np.ndim(value)]
            d.createVariable(var, "f8", dims)[...] = value


def make_grids():
    # Structured: cell (x, y), neighbour (x, y +- 1)
    write(
        "structured.nc",
        NX,
        NY,
        {
            "dx": np.ones((NX, NY, NZ)),
            "dy": np.full((NX, NY, NZ), DY),
            # Uniform spacing (else BOUT++ differentiates dx, which needs x guards)
            "d2x": np.zeros((NX, NY, NZ)),
            "d2y": np.zeros((NX, NY, NZ)),
            "zShift": zshift(X, Y),
            "f_test": F,
        },
    )
    # The same cells, shuffled along x with y of size 1
    rng = np.random.default_rng(3)
    order = rng.permutation(NX * NY)  # order[u] = structured cell number x*NY + y
    position = np.argsort(order)  # position[x*NY + y] = u
    xs, ys = np.divmod(order, NY)
    # Everything but the test field is 2D (x, y): the grid file stays the
    # size of an axisymmetric one
    values = {"f_test": F[xs, ys][:, None, :]}
    # With an FCI-like transform BOUT++ reads the metric in the parallel
    # slices too: the neighbours' values, here the same uniform metric
    uniform = {"dx": 1, "d2x": 0, "d2y": 0, "g_11": 1, "g_22": 1, "g_33": 1}
    uniform |= {"g_13": 0, "g11": 1, "g22": 1, "g33": 1, "g13": 0, "Bxy": 1}
    uniform |= {"dy": DY}
    for var, value in uniform.items():
        values[var] = np.full((NX * NY, 1), float(value))
        if var not in ("dx", "d2x", "d2y"):
            values[f"forward_{var}"] = values[var]
            values[f"backward_{var}"] = values[var]
    for name, step in (("forward", 1), ("backward", -1)):
        yn = (ys + step) % NY
        values[f"pmap_{name}_x"] = position[xs * NY + yn][:, None].astype(float)
        values[f"pmap_{name}_y"] = np.zeros((NX * NY, 1))
        # Toroidal angle by which the field line advances to the neighbour
        values[f"pmap_{name}_dzeta"] = (zshift(xs, yn * DY) - zshift(xs, ys * DY))[
            :, None
        ]
    write("mapped.nc", NX * NY, 1, values)
    return xs, ys


def run(name, nproc, args):
    path = pathlib.Path(name)
    path.mkdir(exist_ok=True)
    shutil.copy("data/BOUT.inp", path)
    status, out = launch_safe(
        f"./test_shiftedmap -d {name} {args}", nproc=nproc, pipe=True
    )
    path.joinpath("run.log").write_text(out)
    assert status == 0, out
    return {
        var: collect(var, path=name, info=False, xguards=False, yguards=False)
        for var in ("f_up", "f_down", "grad_par")
    }


def test_shiftedmap():
    xs, ys = make_grids()
    build_and_log("ShiftedMetricMap")

    structured = run("structured", 1, "mesh:file=structured.nc")
    # Expected slices: the field at the neighbour, at the z the field line reaches
    expected = {}
    for var, step in (("f_up", 1), ("f_down", -1)):
        yn = (Y + step * DY) % (2 * np.pi)
        shift = zshift(X, yn) - zshift(X, Y)
        expected[var] = field(X[..., None], yn[..., None], Z + shift[..., None])
    unshifted = field(X[..., None], (Y + DY)[..., None], Z)

    failures = []
    for var in ("f_up", "f_down"):
        error = np.max(np.abs(structured[var] - expected[var]))
        if error > TOLERANCE:
            failures.append(
                f"ShiftedMetric {var} differs from the field by {error:.2e}"
            )
    size = np.max(np.abs(expected["f_up"] - unshifted))
    if size < 0.1:
        failures.append(f"the shift is too small to test ({size:.2e})")

    for nproc, args in ((1, "NXPE=1"), (2, "NXPE=2")):
        mapped = run(
            f"mapped_np{nproc}",
            nproc,
            f"{args} mesh:file=mapped.nc mesh:paralleltransform:type=shiftedmap",
        )
        for var in ("f_up", "f_down", "grad_par"):
            # Back to the structured layout: mapped cell u is structured (xs[u], ys[u])
            error = np.max(np.abs(mapped[var][:, 0, :] - structured[var][xs, ys]))
            print(f"{nproc} rank(s): {var} differs from ShiftedMetric by {error:.2e}")
            if error > TOLERANCE:
                failures.append(f"{nproc} rank(s): {var} differs by {error:.2e}")
    assert not failures, failures
