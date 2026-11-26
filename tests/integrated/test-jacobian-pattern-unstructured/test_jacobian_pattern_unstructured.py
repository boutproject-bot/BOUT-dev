# Jacobian colouring on an unstructured mesh. The pattern couples each cell to
# the cells that the operators stored in the grid file couple: the parallel
# maps over solver:stencil:parallel_hops steps along the magnetic field, and
# the perpendicular operator.
#
# The model (test_jacobian_pattern_unstructured.cxx) couples cells along the
# field through the support-operator conduction, whose divergence is the
# transpose of its gradient: cells up to two interpolation stencils apart. Its
# finite-difference Jacobian, computed with the colouring of the pattern,
# must therefore be the same with 2 steps as with 6 (which reach every plane
# of this grid), and must differ with 1 step, where couplings are missing.
# The perpendicular operator couples most of the cells that the second step
# adds, so that difference is small (1.7e-5 of the largest entry), but it is
# far above round-off; with 2 steps the Jacobian is the same to the last bit.
#
# Split in y over two processors, the pattern must also hold the couplings
# to cells on the other processor: the Jacobian, with its rows and columns in
# the order of the cells and variables, must be the same as on one.

import shutil
from pathlib import Path

import numpy as np
from boututils.run_wrapper import build_and_log, launch_safe
from read_jacobian import load_jacobian, load_petsc_matrix
from zoidberg.field import Slab
from zoidberg.unstructured import UnstructuredPoloidalGrid, make_unstructured_grid


def make_grid(filename):
    # Tilted field: the field lines cross the irregular plane meshes, so the
    # parallel maps interpolate
    pol = UnstructuredPoloidalGrid.slab(
        8, 8, 1.0, 1.0, R0=1.5, jitter=0.2, jitter_dirs="z", periodic_Z=True, seed=11
    )
    make_unstructured_grid(
        filename, pol, Slab(By=1.0, Bz=0.3, Bzprime=0.0), ny=8, Ly=10.0, k2=False
    )


def jacobian(hops, nproc=1, split=""):
    path = Path(f"data_hops{hops}_np{nproc}")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir()
    shutil.copy("data/BOUT.inp", path / "BOUT.inp")
    status, out = launch_safe(
        f"./test_jacobian_pattern_unstructured -d {path} {split} "
        f"solver:stencil:parallel_hops={hops} solver:save_jacobian_index_base=true",
        nproc=nproc,
        pipe=True,
    )
    (path / "run.log").write_text(out)
    assert status == 0, out
    # The first Jacobian, at the initial state
    first = min(path.glob("jacobian_*.dat"))
    (nrows, _), (indptr, _, _) = load_petsc_matrix(first, "sparse")
    print(
        f"{hops} steps, {nproc} processor(s): "
        f"{indptr[-1] / nrows:.0f} pattern entries per row"
    )
    # Rows and columns in the order of the cells and variables, which does not
    # depend on how the mesh is split between processors
    matrix, _, labels = load_jacobian(path, first)
    order = np.argsort(labels)
    return matrix[np.ix_(order, order)]


def test_jacobian_pattern_unstructured():
    make_grid("grid.fci.nc")
    build_and_log("Jacobian pattern on an unstructured mesh")

    reference = jacobian(6)
    scale = np.max(np.abs(reference))
    two_steps = np.max(np.abs(jacobian(2) - reference)) / scale
    one_step = np.max(np.abs(jacobian(1) - reference)) / scale
    split = jacobian(2, 2, "NYPE=2")
    two_procs = np.max(np.abs(split - reference)) / scale
    print(
        f"Largest difference to 6 steps: {two_steps:.1e} (2 steps), {one_step:.1e} "
        f"(1), {two_procs:.1e} (2 steps on 2 processors)"
    )
    assert two_steps < 1e-12
    assert one_step > 1e-8
    # Split over processors, the right-hand side differs at round-off (sums
    # in another order), which the finite differences magnify (measured 2e-11
    # of the largest entry); a missing coupling would remove whole entries
    assert np.array_equal(split != 0, reference != 0)
    assert two_procs < 1e-9
