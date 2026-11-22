import pathlib
import shutil
import subprocess
import sys

import numpy as np
from boutdata import collect
from boututils.run_wrapper import build_and_log, launch_safe


def analyse_path(path, plot=False, tol=1e-12):

    forward_op = collect("forward_op", path=path)
    forward = collect("forward", path=path)

    grad_par_op = collect("grad_par_op", path=path)
    grad_par_yud = collect("grad_par_yud", path=path)

    div_par_op = collect("div_par_op", path=path)
    div_par_yud = collect("div_par_yud", path=path)

    dV = collect("dV", path=path)  # Cell volume

    # Integrate divergence
    op_div_int = np.sum((dV * div_par_op)[2:-2, :, :])
    yud_div_int = np.sum((dV * div_par_yud)[2:-2, :, :])

    if np.abs(op_div_int) > tol:
        raise ValueError(f"Div_par operator has integral {op_div_int} > {tol}")

    print(f"Div_par volume integral Op: {op_div_int} Yup/down: {yud_div_int}")

    div_par_grad_par_op = collect("div_par_grad_par_op", path=path)
    div_par_grad_par_yud = collect("div_par_grad_par_yud", path=path)

    # Integrate divergence
    op_div2_int = np.sum((dV * div_par_grad_par_op)[2:-2, :, :])
    yud_div2_int = np.sum((dV * div_par_grad_par_yud)[2:-2, :, :])

    if np.abs(op_div2_int) > tol:
        raise ValueError(
            f"Div_par_Grad_par operator has integral {op_div2_int} > {tol}"
        )

    # Operators cached between calls (not rebuilt)
    reused = int(collect("cache_reused", path=path))
    t_first = float(collect("cache_time_first", path=path))
    t_second = float(collect("cache_time_second", path=path))
    print(
        f"Parallel operator cache: reused {reused}, first {t_first:.3g} s, second {t_second:.3g} s"
    )
    if reused != 1:
        raise ValueError("PetscOperators::Cache rebuilt the parallel operators")
    if t_second > 0.01 * t_first:
        raise ValueError(f"Cached request took {t_second} s (first {t_first} s)")

    # Variable-coefficient conduction: the support-operator method conserves
    som = collect("div_par_k_grad_par_som", path=path)
    orig = collect("div_par_k_grad_par_orig", path=path)
    som_int = np.sum((dV * som)[2:-2, :, :])
    orig_int = np.sum((dV * orig)[2:-2, :, :])
    scale = np.sum(np.abs(dV * som)[2:-2, :, :])
    print(
        f"Div_par_K_Grad_par_mod volume integral: petscoperator {som_int:.3e}, original {orig_int:.3e} (scale {scale:.3e})"
    )
    if np.abs(som_int) > tol * max(1.0, scale):
        raise ValueError(
            f"Div_par_K_Grad_par_mod(PetscOperator) has integral {som_int} > {tol}"
        )

    # bndry_flux: no flux through parallel boundaries when false
    nob = collect("div_par_k_grad_par_som_nobndry", path=path)
    wb = collect("div_par_k_grad_par_som_bndry", path=path)
    nob_int = np.sum((dV * nob)[2:-2, :, :])
    wb_int = np.sum((dV * wb)[2:-2, :, :])
    scale = np.sum(np.abs(dV * wb)[2:-2, :, :])
    print(
        f"Dirichlet boundary: integral with bndry_flux=False {nob_int:.3e}, True {wb_int:.3e}"
    )
    if np.abs(nob_int) > tol * max(1.0, scale):
        raise ValueError(f"bndry_flux=False still has boundary flux {nob_int}")
    if np.abs(wb_int) < 1e3 * tol * max(1.0, scale):
        raise ValueError("Test grid has no parallel boundary flux to remove")

    if plot:
        import matplotlib.pyplot as plt

        def plot_comparison(a, alabel, b, blabel):
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            vmax = max([np.amax(a), np.amax(b)])
            vmin = min([np.amin(a), np.amin(b)])

            cs = axs[0].contourf(a, vmin=vmin, vmax=vmax)
            fig.colorbar(cs, ax=axs[0])
            axs[0].set_title(alabel)

            axs[1].contourf(b, vmin=vmin, vmax=vmax)
            axs[1].set_title(blabel)

            cs = axs[2].contourf(a - b)
            axs[2].set_title("Difference")
            fig.colorbar(cs, ax=axs[2])

            return fig

        fig = plot_comparison(
            forward_op[2:-2, 2, :], "Operator", forward[2:-2, 2, :], "Yup/down"
        )
        fig.suptitle("Forward")
        fig.tight_layout()

        fig = plot_comparison(
            grad_par_op[2:-2, 2, :], "Operator", grad_par_yud[2:-2, 2, :], "Yup/down"
        )
        fig.suptitle("Parallel Gradient")
        fig.tight_layout()
        fig.savefig("gradient.pdf")
        fig.savefig("gradient.png")

        fig = plot_comparison(
            div_par_op[2:-2, 2, :],
            f"Operator [integral {op_div_int:.2e}]",
            div_par_yud[2:-2, 2, :],
            f"Yup/down [integral {yud_div_int:.2e}]",
        )
        fig.suptitle("Parallel Divergence")
        fig.tight_layout()
        fig.savefig("divergence.pdf")
        fig.savefig("divergence.png")

        fig = plot_comparison(
            div_par_grad_par_op[2:-2, 2, :],
            f"Operator [integral {op_div2_int:.2e}]",
            div_par_grad_par_yud[2:-2, 2, :],
            f"Yup/down [integral {yud_div2_int:.2e}]",
        )
        fig.suptitle("Parallel diffusion")
        fig.tight_layout()
        fig.savefig("diffusion.pdf")
        fig.savefig("diffusion.png")


def run(nproc, args=""):
    """Run in its own directory (a copy of data/) and return the path."""
    path = f"data_np{nproc}{args.replace(' ', '_').replace('=', '')}"
    pathlib.Path(path).mkdir(exist_ok=True)
    shutil.copy("data/BOUT.inp", path)
    status, out = launch_safe(
        f"./test_petsc_operators -d {path} {args}", nproc=nproc, pipe=True
    )
    pathlib.Path(path, "run.log").write_text(out)
    assert status == 0, out
    return path


# Operator outputs that must not depend on the domain decomposition
DECOMPOSITION_INVARIANT = (
    "forward_op",
    "grad_par_op",
    "div_par_op",
    "div_par_grad_par_op",
    "dV",
)


def check_decomposition(reference, path, rtol=1e-12):
    for name in DECOMPOSITION_INVARIANT:
        a = collect(name, path=reference, info=False)
        b = collect(name, path=path, info=False)
        # Compare away from the X boundaries, as in analyse_path
        np.testing.assert_allclose(
            b[2:-2],
            a[2:-2],
            rtol=rtol,
            atol=rtol * np.max(np.abs(a[2:-2])),
            err_msg=f"{name} differs between {reference} and {path}",
        )


def test_petsc_operators():
    grid_path = "grid.fci.nc"

    # Generate grid file and store in temporary path
    subprocess.check_call([sys.executable, "rotating-ellipse.py", "-o", grid_path])

    # Build executable
    build_and_log("test_petsc_operators")

    reference = run(1)
    analyse_path(reference)

    # The CSR assembly maps stored (grid-file) numbering to PETSc ordering;
    # check that results do not depend on the domain decomposition
    for args in ("NXPE=1", "NXPE=2"):
        path = run(2, args)
        analyse_path(path)
        check_decomposition(reference, path)
