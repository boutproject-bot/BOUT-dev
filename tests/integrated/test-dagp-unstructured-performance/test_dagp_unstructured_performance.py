# Micro-benchmark of the unstructured perpendicular operator
# FV::Div_a_Grad_perp_FCI: time per call and
# per nonzero of its matrices (face gradients, face values, divergence), on
# 1 rank. The bound per nonzero is a loose regression check; the measured
# values are printed to guide optimisation.

import netCDF4
from boutdata.collect import collect
from boututils.run_wrapper import build_and_log, launch_safe
from zoidberg.unstructured import make_test_grid

# Generous bound (about 10x the value measured when the test was written)
MAX_NS_PER_NONZERO = 20.0


def test_dagp_unstructured_performance():
    make_test_grid("bout_mms_cyl", 64, filename="grid.fci.nc", ny=4)
    build_and_log("Perpendicular operator benchmark")

    with netCDF4.Dataset("grid.fci.nc") as grid:
        nnz = {
            name: grid[f"dagp_us_{name}_weights"].shape[0]
            for name in ("gradp", "value", "div")
        }
        cells = int(grid["total_cells"][...])

    status, out = launch_safe(
        "./test_dagp_unstructured_performance", nproc=1, pipe=True
    )
    with open("run.log", "w") as log:
        log.write(out)
    assert status == 0, out

    t = float(collect("time_per_call", path="data", info=False))
    total = sum(nnz.values())
    ns_per_nonzero = 1e9 * t / total
    print(
        f"cells {cells}, nonzeros {nnz} (total {total}, {total / cells:.0f} per cell)"
    )
    print(
        f"time per call {1e3 * t:.3f} ms, {1e9 * t / cells:.1f} ns per cell, "
        f"{ns_per_nonzero:.2f} ns per nonzero"
    )
    assert ns_per_nonzero < MAX_NS_PER_NONZERO
