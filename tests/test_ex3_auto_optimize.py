"""Correctness + speedup tests for Exercise 3 (automatic program optimization)."""
import tempfile

import numpy as np
import pytest

import ex3_auto_optimize as ex3
from mlc_compat import run_tir, time_tir


@pytest.fixture(scope="module")
def ab():
    rng = np.random.default_rng(0)
    a = rng.random((ex3.N, ex3.N), dtype=np.float32)
    b = rng.random((ex3.N, ex3.N), dtype=np.float32)
    return a, b


def test_baseline_matmul_matches_numpy(ab):
    a, b = ab
    out = run_tir(ex3.MatmulModule, [a, b], (ex3.N, ex3.N))
    np.testing.assert_allclose(out, ex3.numpy_matmul(a, b), rtol=1e-2, atol=1e-2)


def test_random_search_is_correct_and_faster(ab):
    a, b = ab
    base = time_tir(ex3.MatmulModule, [a, b], (ex3.N, ex3.N), number=2, repeat=2)
    best_sch, best_t, times = ex3.random_search(ex3.MatmulModule, num_trials=6, seed=1)
    out = run_tir(best_sch.mod, [a, b], (ex3.N, ex3.N))
    np.testing.assert_allclose(out, ex3.numpy_matmul(a, b), rtol=1e-2, atol=1e-2)
    assert len(times) == 6
    # A tiled/parallel/vectorized schedule must beat the naive triple loop.
    assert best_t < base


@pytest.mark.slow
def test_meta_schedule_tunes_a_faster_program(ab):
    import shutil

    a, b = ab
    base = time_tir(ex3.MatmulModule, [a, b], (ex3.N, ex3.N), number=2, repeat=2)
    # NB: don't use TemporaryDirectory's auto-cleanup -- meta-schedule leaves
    # log files that trip Windows' shutil.rmtree; clean up defensively instead.
    wd = tempfile.mkdtemp(prefix="ms_test_")
    try:
        sch = ex3.tune_with_meta_schedule(
            ex3.MatmulModule, wd, max_trials=12, inputs=[a, b], out_shape=(ex3.N, ex3.N)
        )
        out = run_tir(sch.mod, [a, b], (ex3.N, ex3.N))
        np.testing.assert_allclose(out, ex3.numpy_matmul(a, b), rtol=1e-2, atol=1e-2)
        tuned = time_tir(sch.mod, [a, b], (ex3.N, ex3.N), number=3, repeat=2)
        assert tuned < base
    finally:
        shutil.rmtree(wd, ignore_errors=True)
