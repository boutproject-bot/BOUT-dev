#!/usr/bin/env python3
#
# Python script to run and analyse MMS test


import os

import numpy as np
import pytest
from boutdata.collect import collect as _collect
from boututils.datafile import DataFile
from boututils.run_wrapper import launch_safe
from numpy.testing import assert_allclose

if not os.path.exists(os.path.join(os.path.dirname(__file__), "grid.fci.nc")):
    pytest.skip(
        "grid.fci.nc not found (Zoidberg likely missing), skipping test.",
        allow_module_level=True,
    )

directory = "data"


def collect(*args):
    return _collect(
        *args,
        info=False,
        path=directory,
        xguards=False,
        yguards=False,
    )


def test_fci_boundary():

    nprocs = [1]
    mthread = 2

    with DataFile("grid.fci.nc") as grid:
        MXG = grid.get("MXG", default=1)
        xfwd = grid.read("forward_xt_prime")[MXG:-MXG]
        xbwd = grid.read("backward_xt_prime")[MXG:-MXG]

    nx = xfwd.shape[0]

    regions = {
        "xin_fwd": xfwd < MXG,
        "xout_fwd": xfwd > nx + MXG - 1,
        "xin_bwd": xbwd < MXG,
        "xout_bwd": xbwd > nx + MXG - 1,
    }
    regions = {k: v.astype(int) for k, v in regions.items()}

    for x in "xout", "xin":
        regions[x] = regions[f"{x}_fwd"] + regions[f"{x}_bwd"]
    for x in "fwd", "bwd":
        regions[x] = regions[f"xin_{x}"] + regions[f"xout_{x}"]
    regions["all"] = regions["xin"] + regions["xout"]

    bndrys = {
        "ybndry_-1": regions["xout_bwd"],
        "ybndry_0": regions["xout_fwd"] * 0,
        "ybndry_1": regions["xout_fwd"],
    }

    for nproc in nprocs:
        cmd = "./get_par_bndry"
        _, _out = launch_safe(cmd, nproc=nproc, mthread=mthread, pipe=True)

    for k, v in regions.items():
        data = collect(f"field_{k}")
        assert_allclose(data, v)
    for i in range(-1, 2):
        name = f"ybndry_{i}"
        data = collect(name)
        assert_allclose(bndrys[name], data)

    # Points with a boundary in both directions have no valid neighbouring
    # point (MYG = 1: |(-1) - (+1)| - 2 = 0); all others have one
    both = (regions["fwd"] > 0) & (regions["bwd"] > 0)
    assert np.any(both), "The grid should have doubly bounded points"
    for k in ("xin_fwd", "xout_fwd", "xin_bwd", "xout_bwd"):
        valid = collect(f"valid_{k}")
        here = regions[k] > 0
        expected = np.where(both, 0, 1)
        assert_allclose(valid[here], expected[here], err_msg=k)
        assert np.all(valid[~here] == -1), k
