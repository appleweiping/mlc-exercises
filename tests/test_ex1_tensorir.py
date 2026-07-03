"""Numerical-correctness tests for Exercise 1 (TensorIR)."""
import numpy as np

import ex1_tensorir as ex1
from mlc_compat import run_tir


def test_tvmscript_matmul_relu_matches_numpy():
    a = np.random.rand(128, 128).astype("float32")
    b = np.random.rand(128, 128).astype("float32")
    out = run_tir(ex1.MatmulReLU, [a, b], (128, 128))
    np.testing.assert_allclose(out, ex1.numpy_mm_relu(a, b), rtol=1e-4, atol=1e-4)


def test_te_matmul_relu_matches_numpy():
    a = np.random.rand(128, 128).astype("float32")
    b = np.random.rand(128, 128).astype("float32")
    mod = ex1.te_matmul_relu(128)
    out = run_tir(mod, [a, b], (128, 128))
    np.testing.assert_allclose(out, ex1.numpy_mm_relu(a, b), rtol=1e-4, atol=1e-4)


def test_scheduled_matmul_relu_preserves_numerics():
    a = np.random.rand(128, 128).astype("float32")
    b = np.random.rand(128, 128).astype("float32")
    sch = ex1.schedule_matmul_relu(ex1.MatmulReLU)
    out = run_tir(sch.mod, [a, b], (128, 128))
    np.testing.assert_allclose(out, ex1.numpy_mm_relu(a, b), rtol=1e-4, atol=1e-4)


def test_schedule_actually_transformed_the_loops():
    """The scheduled module must differ from the original (real transformation)."""
    before = ex1.MatmulReLU["main"].script()
    sch = ex1.schedule_matmul_relu(ex1.MatmulReLU)
    after = sch.mod["main"].script()
    assert before != after
    # Evidence the intended primitives took effect.
    assert "T.parallel" in after
    assert "T.vectorized" in after


def test_bmm_bias_relu_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.random((4, 64, 64), dtype=np.float32)
    w = rng.random((4, 64, 64), dtype=np.float32)
    bias = rng.random((64,), dtype=np.float32)
    mod = ex1.te_bmm_bias_relu()
    out = run_tir(mod, [x, w, bias], (4, 64, 64))
    np.testing.assert_allclose(out, ex1.numpy_bmm_bias_relu(x, w, bias), rtol=1e-3, atol=1e-3)


def test_scheduled_bmm_preserves_numerics():
    rng = np.random.default_rng(1)
    x = rng.random((4, 64, 64), dtype=np.float32)
    w = rng.random((4, 64, 64), dtype=np.float32)
    bias = rng.random((64,), dtype=np.float32)
    mod = ex1.te_bmm_bias_relu()
    sch = ex1.schedule_bmm(mod)
    out = run_tir(sch.mod, [x, w, bias], (4, 64, 64))
    np.testing.assert_allclose(out, ex1.numpy_bmm_bias_relu(x, w, bias), rtol=1e-3, atol=1e-3)
