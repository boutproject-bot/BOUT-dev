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
#
# Single null: a tokamak-like grid with a core (closed field lines, twist
# at the branch cut), a private flux region below the X-point and a scrape
# off layer, recast as above with the neighbours of BOUT++'s single-null
# topology and both targets marked as outer. ShiftedMetric on the
# structured grid and the map must both match the field at the shifted
# points, and the map the boundary condition at the targets.

import pathlib
import shutil

import numpy as np
from boutdata.collect import collect
from boututils.run_wrapper import build_and_log, launch, launch_safe
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


def write(name, nx, ny, values, shift_angle, topology=None):
    with Dataset(name, "w") as d:
        d.createDimension("x", nx)
        d.createDimension("y", ny)
        d.createDimension("z", NZ)
        integers = {"nx": nx, "ny": ny, "nz": NZ} | (topology or {})
        for var, value in integers.items():
            d.createVariable(var, "i4")[...] = value
        d.createVariable("ShiftAngle", "f8", ("x",))[...] = shift_angle
        for var, value in values.items():
            dims = ("x", "y", "z")[: np.ndim(value)]
            d.createVariable(var, "f8", dims)[...] = value


def shuffle(ny, seed):
    """A random order of the NX * ny cells along x: cell number x * ny + y
    is stored at position[x * ny + y], and xs, ys are the structured
    indices of the stored cells"""
    order = np.random.default_rng(seed).permutation(NX * ny)
    xs, ys = np.divmod(order, ny)
    return np.argsort(order), xs, ys


def uniform_metric(ncells, dy):
    """A uniform metric for cells stored along x, with y of size 1. With an
    FCI-like transform BOUT++ also reads the metric in the parallel slices
    (the neighbours' values). Everything is 2D (x, y): the grid file stays
    the size of an axisymmetric one"""
    uniform = {"dx": 1, "d2x": 0, "d2y": 0, "g_11": 1, "g_22": 1, "g_33": 1}
    uniform |= {"g_13": 0, "g11": 1, "g22": 1, "g33": 1, "g13": 0, "Bxy": 1}
    uniform |= {"dy": dy}
    values = {}
    for var, value in uniform.items():
        values[var] = np.full((ncells, 1), float(value))
        if var not in ("dx", "d2x", "d2y"):
            values[f"forward_{var}"] = values[var]
            values[f"backward_{var}"] = values[var]
    return values


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
    position, xs, ys = shuffle(NY, seed=3)
    values = uniform_metric(NX * NY, DY) | {"f_test": F[xs, ys][:, None, :]}
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

    # The area of a y face is taken from the cell on one side, so a cell
    # whose dx differs from its neighbours' must be refused
    shutil.copy("mapped.nc", "mapped_bad_dx.nc")
    with Dataset("mapped_bad_dx.nc", "a") as d:
        d["dx"][0, 0] = 1.5 * d["dx"][0, 0]
    path = pathlib.Path("mapped_bad_dx")
    path.mkdir(exist_ok=True)
    shutil.copy("data/BOUT.inp", path)
    status, out = launch(
        f"./test_shiftedmap -d {path} mesh:file=mapped_bad_dx.nc "
        "mesh:paralleltransform:type=shiftedmap",
        nproc=1,
        pipe=True,
    )
    if status == 0 or "the y faces need the same dx" not in out:
        failures.append("a dx that differs between neighbours was not refused")
    assert not failures, failures


# Single null, 4 x 12 cells: inside the separatrix (x < IXSEPS1) the inner
# leg (y <= JYSEPS1_1) and the outer leg (y > JYSEPS2_2) form the private
# flux region below the X-point, and the core in between is closed, with a
# twist at its branch cut. BOUT++ connects these regions only at processor
# boundaries in y, so ShiftedMetric runs on SN_NYPE processors in y.
SN_NY, SN_NYPE = 12, 3
IXSEPS1, JYSEPS1_1, JYSEPS2_2 = 2, 3, 7
SN_DY = 2 * np.pi / SN_NY


def sn_zshift(x, j):
    return 0.4 * (x + 1) * j + 0.7 * np.sin(j * SN_DY + x)


def single_null_neighbours(step):
    """For each cell (x, y): the y of its neighbour along y + step in
    BOUT++'s single-null topology (-1 at a target), and the toroidal angle
    the field line advances to it"""
    x, j = np.meshgrid(np.arange(NX), np.arange(SN_NY), indexing="ij")
    inside = x < IXSEPS1
    core = inside & (j > JYSEPS1_1) & (j <= JYSEPS2_2)
    yn = j + step
    if step == 1:
        cut = core & (j == JYSEPS2_2)
        yn[cut] = JYSEPS1_1 + 1
        yn[inside & (j == JYSEPS1_1)] = JYSEPS2_2 + 1  # inner leg to outer leg
    else:
        cut = core & (j == JYSEPS1_1 + 1)
        yn[cut] = JYSEPS2_2
        yn[inside & (j == JYSEPS2_2 + 1)] = JYSEPS1_1  # outer leg to inner leg
    target = (yn < 0) | (yn >= SN_NY)
    yn = np.where(target, -1, yn)
    twist = np.where(cut, step * 2 * np.pi * q(x), 0.0)
    dzeta = sn_zshift(x, yn) - sn_zshift(x, j) + twist
    return yn, np.where(target, 0.0, dzeta)


def make_single_null():
    x, j = np.meshgrid(np.arange(NX), np.arange(SN_NY), indexing="ij")
    f = field(x[..., None], j[..., None] * SN_DY, Z)
    topology = {"ixseps1": IXSEPS1, "ixseps2": NX, "jyseps1_1": JYSEPS1_1}
    topology |= {"jyseps2_1": SN_NY // 2, "jyseps1_2": SN_NY // 2}
    topology |= {"jyseps2_2": JYSEPS2_2, "ny_inner": SN_NY // 2}
    write(
        "single_null.nc",
        NX,
        SN_NY,
        {
            "dx": np.ones(x.shape),
            "dy": np.full(x.shape, SN_DY),
            "d2x": np.zeros(x.shape),
            "d2y": np.zeros(x.shape),
            "zShift": sn_zshift(x, j),
            "f_test": f,
        },
        2 * np.pi * q(np.arange(NX)),
        topology,
    )
    ncells = NX * SN_NY
    position, xs, ys = shuffle(SN_NY, seed=5)
    values = uniform_metric(ncells, SN_DY) | {"f_test": f[xs, ys][:, None, :]}
    for name, step in (("forward", 1), ("backward", -1)):
        yn, dzeta = (a[xs, ys] for a in single_null_neighbours(step))
        # Both targets outer, so that the default y boundary iterator visits them
        neighbour = np.where(yn < 0, ncells, position[xs * SN_NY + yn])
        values[f"pmap_{name}_x"] = neighbour[:, None]
        values[f"pmap_{name}_y"] = np.zeros((ncells, 1))
        values[f"pmap_{name}_dzeta"] = dzeta[:, None]
    write("single_null_mapped.nc", ncells, 1, values, np.zeros(ncells))
    return f, xs, ys


def test_shiftedmap_single_null():
    f, xs, ys = make_single_null()
    build_and_log("ShiftedMetricMap")

    x = np.broadcast_to(np.arange(NX)[:, None], (NX, SN_NY))
    expected = {}
    targets = {}
    for var, step in (("f_up", 1), ("f_down", -1)):
        yn, dzeta = single_null_neighbours(step)
        yn_angle = (np.maximum(yn, 0) * SN_DY)[..., None]
        expected[var] = field(x[..., None], yn_angle, Z + dzeta[..., None])
        targets[var] = (yn < 0)[..., None]

    failures = []
    structured = run("single_null", SN_NYPE, f"NYPE={SN_NYPE} mesh:file=single_null.nc")
    for var in ("f_up", "f_down"):
        # Away from the targets, where ShiftedMetric's slices use y guard cells
        difference = np.abs(structured[var] - expected[var])
        error = np.max(np.where(targets[var], 0.0, difference))
        print(f"single null, ShiftedMetric: {var} differs by {error:.2e}")
        if error > TOLERANCE:
            failures.append(f"ShiftedMetric {var} differs by {error:.2e}")

    # On the map, the third-order Dirichlet condition at the targets
    along = dict(expected)
    for var, other in (("f_up", "f_down"), ("f_down", "f_up")):
        o3 = 8 / 3 * TARGET_VALUE - 2 * f + along[other] / 3
        expected[var] = np.where(targets[var], o3, along[var])
    expected["grad_par"] = (expected["f_up"] - expected["f_down"]) / (2 * SN_DY)
    points = NX * NZ
    expected_points = {"points_xin_fwd": 0, "points_xin_bwd": 0}
    expected_points |= {"points_xout_fwd": points, "points_xout_bwd": points}
    expected_points |= {"ybndry_up": points, "ybndry_down": points}
    for nproc in (1, 2):
        mapped = run(
            f"single_null_mapped_np{nproc}",
            nproc,
            f"NXPE={nproc} mesh:file=single_null_mapped.nc "
            "mesh:paralleltransform:type=shiftedmap",
        )
        label = f"single null map, {nproc} rank(s)"
        for var in ("f_up", "f_down", "grad_par"):
            error = np.max(np.abs(mapped[var][:, 0, :] - expected[var][xs, ys]))
            print(f"{label}: {var} differs by {error:.2e}")
            if error > TOLERANCE:
                failures.append(f"{label}: {var} differs by {error:.2e}")
        found = {var: mapped[var] for var in COUNTS}
        if found != expected_points:
            failures.append(f"{label}: target points {found}, not {expected_points}")
    assert not failures, failures
