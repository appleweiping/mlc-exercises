"""Correctness tests for Exercise 2 (end-to-end Relax model, MLC assignment 1)."""
import os
import pickle

import numpy as np
import pytest

import ex2_end_to_end as ex2
from mlc_compat import build_relax_vm, cpu, to_tvm

_PARAMS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "fasionmnist_mlp_assignment_params.pkl",
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(_PARAMS),
    reason="run scripts/get_data.py first to download assignment weights",
)


@pytest.fixture(scope="module")
def weight_map():
    with open(_PARAMS, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def sample_input():
    rng = np.random.default_rng(0)
    return rng.random(ex2.INPUT_SHAPE, dtype=np.float32)


def _run_vm(mod, x):
    vm = build_relax_vm(mod)
    out = vm["main"](to_tvm(x, cpu()))
    return out.numpy()


def test_emit_te_model_matches_reference(weight_map, sample_input):
    mod = ex2.create_model_via_emit_te(weight_map)
    out = _run_vm(mod, sample_input)
    ref = ex2.numpy_reference(sample_input, weight_map)
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)


def test_torch_vendor_func_model_matches_reference(weight_map, sample_input):
    mod = ex2.create_model_with_torch_func(weight_map)
    out = _run_vm(mod, sample_input)
    ref = ex2.numpy_reference(sample_input, weight_map)
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)


def test_scheduled_conv2d_preserves_output(weight_map, sample_input):
    mod = ex2.create_model_via_emit_te(weight_map)
    ref = _run_vm(mod, sample_input)
    sched_mod = ex2.schedule_conv2d(mod)
    out = _run_vm(sched_mod, sample_input)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_output_is_a_probability_distribution(weight_map, sample_input):
    mod = ex2.create_model_via_emit_te(weight_map)
    out = _run_vm(mod, sample_input)
    assert out.shape == (ex2.BATCH, 10)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(ex2.BATCH), rtol=1e-4, atol=1e-4)
    assert (out >= -1e-6).all()
