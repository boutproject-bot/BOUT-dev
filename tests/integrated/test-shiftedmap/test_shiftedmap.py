#!/usr/bin/env python3
#
# ShiftedMetricMap (paralleltransform type = shiftedmap): the neighbour of
# each cell along the magnetic field comes from a map in the grid file and
# the toroidal shift to it is applied with FFTs in z.
#
# A sheared slab with closed field lines (4 x 8 cells, nz = 16, zShift
# varying in x and y, with a twist-shift at the branch cut) is run with
# ShiftedMetric, and the same cells are stored in a shuffled order along x
# (y of size 1) with the map and run with ShiftedMetricMap, on one and on
# two processors. For every cell the forward and backward parallel slices
# and Grad_par must agree to round-off, and the slices must match the field
# evaluated at the shifted points. The shift, and the twist at the cut,
# each change the slices by order one.
#
# Open field lines: the same cells with targets after the last and before
# the first y (marked as outer going forward, as inner going backward) and
# one cell with targets on both sides. A parallel Dirichlet condition of
# third order sets the slices at the targets; they must match the
# extrapolation through the boundary value half a cell away, of second
# order for the cell with two targets, and the target points must be in
# the expected boundary regions.

import pathlib
import shutil

import numpy as np
from boutdata.collect import collect
from boututils.run_wrapper import build_and_log, launch_safe
from netCDF4 import Dataset

NX, NY, NZ = 4, 8, 16
TOLERANCE = 1e-12
DY = 2 * np.pi / NY
TARGET_VALUE = 0.3  # parallel_dirichlet_o3(0.3) in test_shiftedmap.cxx
TWO_TARGETS = (1, 3)  # (x, y) of the cell with targets on both sides
COUNTS = ("points_xin_fwd", "points_xin_bwd", "points_xout_fwd", "points_xout_bwd")
COUNTS += ("ybndry_up", "ybndry_down")


def field(x, y, z):
    """Band-limited in z (modes 2 and 5 of 8), so that FFT shifts are exact"""
    return np.sin(2 * z + 0.9 * y + x) + 0.4 * np.cos(5 * z - 2 * y + 0.3 * x)


def q(x):
    """Toroidal turns of a field line per poloidal turn"""
    return 0.3 + 0.1 * x


def zshift(x, y):
    """Sheared, and not periodic: it grows by 2 pi q(x) per poloidal turn"""
    return q(x) * y + 0.7 * (x + 1) * np.sin(y)


# Cell indices; along a field line y continues past the branch cut
X, J = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
Z = np.arange(NZ) * 2 * np.pi / NZ
F = field(X[..., None], J[..., None] * DY, Z)


def write(name, nx, ny, values, shift_angle):
    with Dataset(name, "w") as d:
        d.createDimension("x", nx)
        d.createDimension("y", ny)
        d.createDimension("z", NZ)
        for var, value in (("nx", nx), ("ny", ny), ("nz", NZ)):
            d.createVariable(var, "i4")[...] = value
        d.createVariable("ShiftAngle", "f8", ("x",))[...] = shift_angle
        for var, value in values.items():
            dims = ("x", "y", "z")[: np.ndim(value)]
            d.createVariable(var, "f8", dims)[...] = value


def make_grids():
    # Structured: cell (x, y), neighbour (x, y +- 1); zShift jumps by
    # ShiftAngle at the branch cut
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
            "zShift": zshift(X, J * DY),
            "f_test": F,
        },
        2 * np.pi * q(np.arange(NX)),
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
        values[f"pmap_{name}_x"] = position[xs * NY + (ys + step) % NY][:, None]
        values[f"pmap_{name}_y"] = np.zeros((NX * NY, 1))
        # Toroidal angle by which the field line advances to the neighbour,
        # along the line: across the branch cut it includes the twist-shift
        dzeta = zshift(xs, (ys + step) * DY) - zshift(xs, ys * DY)
        values[f"pmap_{name}_dzeta"] = dzeta[:, None]
    # No branch cut in y here: the twist is part of the map
    write("mapped.nc", NX * NY, 1, values, np.zeros(NX * NY))

    # Open field lines: targets beyond the last and the first y, marked as
    # outer (nx) going forward and as inner (-1) going backward
    two_targets = (xs == TWO_TARGETS[0]) & (ys == TWO_TARGETS[1])
    for name, step, mark in (("forward", 1, NX * NY), ("backward", -1, -1)):
        ends = (ys + step < 0) | (ys + step >= NY) | two_targets
        pmap = values[f"pmap_{name}_x"]
        values[f"pmap_{name}_x"] = np.where(ends[:, None], mark, pmap)
    write("mapped_open.nc", NX * NY, 1, values, np.zeros(NX * NY))
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
    result = {
        var: collect(var, path=name, info=False, xguards=False, yguards=False)
        for var in ("f_up", "f_down", "grad_par")
    }
    return result | {var: int(collect(var, path=name, info=False)) for var in COUNTS}


def open_field_lines(closed):
    """Expected slices on open field lines, from those on closed ones"""
    ends = {1: J == NY - 1, -1: J == 0}
    two_targets = (X == TWO_TARGETS[0]) & (J == TWO_TARGETS[1])
    result = {}
    for var, other, step in (("f_up", "f_down", 1), ("f_down", "f_up", -1)):
        # Quadratic through the target value half a cell away, the cell, and
        # the cell's other neighbour; linear without that neighbour
        o3 = 8 / 3 * TARGET_VALUE - 2 * F + closed[other] / 3
        o2 = 2 * TARGET_VALUE - F
        result[var] = np.where(ends[step][..., None], o3, closed[var])
        result[var] = np.where(two_targets[..., None], o2, result[var])
    # The cells next to the cell with two targets still see it
    result["grad_par"] = (result["f_up"] - result["f_down"]) / (2 * DY)
    return result


def slice_at(step, dzeta):
    """The field in the cell at y + step, at z + dzeta"""
    yn = ((J + step) % NY) * DY
    return field(X[..., None], yn[..., None], Z + dzeta[..., None])


def test_shiftedmap():
    xs, ys = make_grids()
    build_and_log("ShiftedMetricMap")

    structured = run("structured", 1, "mesh:file=structured.nc")
    # Expected slices: the field at the neighbour, at the z the field line reaches
    expected = {}
    for var, step in (("f_up", 1), ("f_down", -1)):
        expected[var] = slice_at(step, zshift(X, (J + step) * DY) - zshift(X, J * DY))

    failures = []
    for var in ("f_up", "f_down"):
        error = np.max(np.abs(structured[var] - expected[var]))
        if error > TOLERANCE:
            failures.append(
                f"ShiftedMetric {var} differs from the field by {error:.2e}"
            )
    # Neither the shift nor the twist at the cut is small
    unshifted = slice_at(1, np.zeros((NX, NY)))
    untwisted = slice_at(1, zshift(X, ((J + 1) % NY) * DY) - zshift(X, J * DY))
    for name, other in (("shift", unshifted), ("twist", untwisted)):
        size = np.max(np.abs(expected["f_up"] - other))
        print(f"the {name} changes the slices by up to {size:.2f}")
        if size < 0.1:
            failures.append(f"the {name} is too small to test ({size:.2e})")

    # Target points: none on closed field lines. On open ones NZ per target
    # cell, forward ones outer and backward ones inner; the default y
    # boundary iterator visits the outer ones
    none = dict.fromkeys(COUNTS, 0)
    targets = (NX + 1) * NZ
    open_points = none | {"points_xout_fwd": targets, "points_xin_bwd": targets}
    open_points |= {"ybndry_up": targets}
    expected_open = open_field_lines(expected)
    size = np.max(np.abs(expected_open["f_up"] - expected["f_up"]))
    print(f"the targets change the slices by up to {size:.2f}")
    if size < 0.1:
        failures.append(f"the targets are too small to test ({size:.2e})")

    def check_points(label, result, expected_points):
        points = {var: result[var] for var in COUNTS}
        if points != expected_points:
            failures.append(f"{label}: target points {points}, not {expected_points}")

    check_points("ShiftedMetric", structured, none)
    for nproc, args in ((1, "NXPE=1"), (2, "NXPE=2")):
        for grid, reference, points in (
            ("mapped", structured, none),
            ("mapped_open", expected_open, open_points),
        ):
            mapped = run(
                f"{grid}_np{nproc}",
                nproc,
                f"{args} mesh:file={grid}.nc mesh:paralleltransform:type=shiftedmap",
            )
            label = f"{grid}, {nproc} rank(s)"
            for var in ("f_up", "f_down", "grad_par"):
                # Back to the structured layout: mapped cell u is (xs[u], ys[u])
                error = np.max(np.abs(mapped[var][:, 0, :] - reference[var][xs, ys]))
                print(f"{label}: {var} differs by {error:.2e}")
                if error > TOLERANCE:
                    failures.append(f"{label}: {var} differs by {error:.2e}")
            check_points(label, mapped, points)
    assert not failures, failures
