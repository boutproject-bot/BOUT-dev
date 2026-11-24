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

import shutil
from pathlib import Path

import numpy as np
from boututils.run_wrapper import build_and_log, launch_safe
from read_jacobian import load_petsc_matrix
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


def jacobian(hops):
    path = Path(f"data_hops{hops}")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir()
    shutil.copy("data/BOUT.inp", path / "BOUT.inp")
    status, out = launch_safe(
        f"./test_jacobian_pattern_unstructured -d {path} "
        f"solver:stencil:parallel_hops={hops}",
        nproc=1,
        pipe=True,
    )
    (path / "run.log").write_text(out)
    assert status == 0, out
    # The first Jacobian, at the initial state
    first = min(path.glob("jacobian_*.dat"))
    (nrows, _), (indptr, _, _) = load_petsc_matrix(first, "sparse")
    print(f"{hops} steps: {indptr[-1] / nrows:.0f} pattern entries per row")
    return load_petsc_matrix(first)


def test_jacobian_pattern_unstructured():
    make_grid("grid.fci.nc")
    build_and_log("Jacobian pattern on an unstructured mesh")

    reference = jacobian(6)
    scale = np.max(np.abs(reference))
    two_steps = np.max(np.abs(jacobian(2) - reference)) / scale
    one_step = np.max(np.abs(jacobian(1) - reference)) / scale
    print(
        f"Largest difference to 6 steps: {two_steps:.1e} (2 steps), {one_step:.1e} (1)"
    )
    assert two_steps < 1e-12
    assert one_step > 1e-8
