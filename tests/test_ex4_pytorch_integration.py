"""Correctness tests for Exercise 4 (PyTorch -> Relax integration, lecture 6)."""
import numpy as np

import ex4_pytorch_integration as ex4


def test_tiny_mlp_import_matches_pytorch():
    torch_import_check(ex4.TinyMLP(), (4, 16))


def test_small_cnn_import_matches_pytorch():
    torch_import_check(ex4.SmallCNN(), (2, 1, 28, 28))


def test_optimize_pipeline_preserves_numerics():
    model = ex4.SmallCNN().eval()
    x = np.random.rand(2, 1, 28, 28).astype("float32")
    mod = ex4.import_torch_model(model, (2, 1, 28, 28))
    out_opt = ex4.build_and_run(mod, x, optimize=True)
    out_plain = ex4.build_and_run(mod, x, optimize=False)
    np.testing.assert_allclose(out_opt, out_plain, rtol=1e-5, atol=1e-6)


def torch_import_check(model, shape):
    x = np.random.rand(*shape).astype("float32")
    ref = ex4.torch_reference(model, x)
    mod = ex4.import_torch_model(model, shape)
    out = ex4.build_and_run(mod, x)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)
