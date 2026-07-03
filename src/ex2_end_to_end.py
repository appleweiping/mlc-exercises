"""MLC Exercise 2 -- End-to-End Model Execution in Relax.

This is the core of the official MLC *assignment 1* (sections 2-4), grounded on
``mlc-ai/notebooks/assignment/assignment1.ipynb`` and lecture 4
(*Build End-to-End Model*).

The model is the assignment's Fashion-MNIST classifier::

    Conv2d(1->32, 3x3) + bias -> ReLU -> MaxPool2d(2x2) -> Flatten
        -> Linear(5408->100) + bias -> ReLU -> Linear(100->10) + bias -> Softmax

Three constructions are implemented, each returning a Relax ``IRModule`` that
runs on the Relax VirtualMachine and matches a NumPy/PyTorch reference:

1. ``create_model_via_emit_te`` -- build the whole graph with
   ``relax.BlockBuilder`` + ``emit_te`` using TOPI operators (assignment sec. 2).
2. ``create_model_with_torch_func`` -- replace the conv2d with a registered
   external PyTorch kernel called through ``call_dps_packed`` (assignment sec. 3,
   "use of vendor library").
3. ``schedule_conv2d`` -- take the TensorIR conv2d produced in (1) and transform
   it with ``compute_inline`` / ``fuse`` / ``parallel`` / ``vectorize``
   (assignment sec. 4, "transformation in end-to-end models").

Weights are loaded from the assignment's own pre-trained parameter pickle
(downloaded by ``scripts/get_data.py``; ~83% test accuracy on Fashion-MNIST).
"""

from __future__ import annotations

import numpy as np
import tvm
from tvm import relax, topi
from tvm.relax import BlockBuilder

BATCH = 4
INPUT_SHAPE = (BATCH, 1, 28, 28)  # NCHW


# ---------------------------------------------------------------------------
# TOPI-based Tensor Expression building blocks for emit_te
# ---------------------------------------------------------------------------
def _te_conv2d(x, w):
    # NCHW conv, valid padding, unit stride -> (N, 32, 26, 26)
    return topi.nn.conv2d_nchw(x, w, stride=1, padding=0, dilation=1, out_dtype="float32")


def _te_bias_add_nchw(x, b):
    # b already reshaped to (1, C, 1, 1)
    return topi.add(x, b)


def _te_relu(x):
    return topi.nn.relu(x)


def _te_maxpool(x):
    return topi.nn.pool2d(x, [2, 2], [2, 2], [1, 1], [0, 0, 0, 0], "max", layout="NCHW")


def _te_flatten(x):
    return topi.nn.flatten(x)


def _te_dense_bias(x, w, b):
    # w: (out, in), b: (1, out)
    y = topi.nn.dense(x, w)
    return topi.add(y, b)


def _te_softmax(x):
    return topi.nn.softmax(x)


# ---------------------------------------------------------------------------
# 1. Build the end-to-end model via BlockBuilder + emit_te (assignment sec. 2)
# ---------------------------------------------------------------------------
def create_model_via_emit_te(weight_map: dict) -> tvm.IRModule:
    """Construct the Fashion-MNIST classifier as a Relax IRModule."""
    bb = BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo(INPUT_SHAPE, "float32"))

    conv2d_weight = relax.const(weight_map["conv2d_weight"], "float32")
    conv2d_bias = relax.const(weight_map["conv2d_bias"].reshape(1, 32, 1, 1), "float32")
    linear0_weight = relax.const(weight_map["linear0_weight"], "float32")
    linear0_bias = relax.const(weight_map["linear0_bias"].reshape(1, 100), "float32")
    linear1_weight = relax.const(weight_map["linear1_weight"], "float32")
    linear1_bias = relax.const(weight_map["linear1_bias"].reshape(1, 10), "float32")

    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(_te_conv2d, x, conv2d_weight)
            lv1 = bb.emit_te(_te_bias_add_nchw, lv0, conv2d_bias)
            lv2 = bb.emit_te(_te_relu, lv1)
            lv3 = bb.emit_te(_te_maxpool, lv2)
            lv4 = bb.emit_te(_te_flatten, lv3)
            lv5 = bb.emit_te(_te_dense_bias, lv4, linear0_weight, linear0_bias)
            lv6 = bb.emit_te(_te_relu, lv5)
            lv7 = bb.emit_te(_te_dense_bias, lv6, linear1_weight, linear1_bias)
            lv8 = bb.emit_te(_te_softmax, lv7)
            gv = bb.emit_output(lv8)
        bb.emit_func_output(gv)
    return bb.get()


# ---------------------------------------------------------------------------
# 2. Vendor-library integration: external PyTorch conv2d (assignment sec. 3)
# ---------------------------------------------------------------------------
def register_torch_conv2d():
    """Register an external runtime function that runs conv2d with PyTorch."""
    import torch

    @tvm.register_global_func("env.conv2d", override=True)
    def torch_conv2d(x, w, out):  # pragma: no cover - exercised via VM
        x_t = torch.from_numpy(x.numpy())
        w_t = torch.from_numpy(w.numpy())
        res = torch.nn.functional.conv2d(x_t, w_t)
        out.copyfrom(res.detach().numpy())


def create_model_with_torch_func(weight_map: dict) -> tvm.IRModule:
    """Same model, but conv2d is delegated to the registered torch kernel via
    ``call_dps_packed`` (destination-passing external call)."""
    register_torch_conv2d()
    bb = BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo(INPUT_SHAPE, "float32"))

    conv2d_weight = relax.const(weight_map["conv2d_weight"], "float32")
    conv2d_bias = relax.const(weight_map["conv2d_bias"].reshape(1, 32, 1, 1), "float32")
    linear0_weight = relax.const(weight_map["linear0_weight"], "float32")
    linear0_bias = relax.const(weight_map["linear0_bias"].reshape(1, 100), "float32")
    linear1_weight = relax.const(weight_map["linear1_weight"], "float32")
    linear1_bias = relax.const(weight_map["linear1_bias"].reshape(1, 10), "float32")

    conv_out_sinfo = relax.TensorStructInfo((BATCH, 32, 26, 26), "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(
                relax.call_dps_packed(
                    "env.conv2d", (x, conv2d_weight), out_sinfo=conv_out_sinfo
                )
            )
            lv1 = bb.emit_te(_te_bias_add_nchw, lv0, conv2d_bias)
            lv2 = bb.emit_te(_te_relu, lv1)
            lv3 = bb.emit_te(_te_maxpool, lv2)
            lv4 = bb.emit_te(_te_flatten, lv3)
            lv5 = bb.emit_te(_te_dense_bias, lv4, linear0_weight, linear0_bias)
            lv6 = bb.emit_te(_te_relu, lv5)
            lv7 = bb.emit_te(_te_dense_bias, lv6, linear1_weight, linear1_bias)
            lv8 = bb.emit_te(_te_softmax, lv7)
            gv = bb.emit_output(lv8)
        bb.emit_func_output(gv)
    return bb.get()


# ---------------------------------------------------------------------------
# 3. Transform the conv2d TensorIR inside the end-to-end model (assignment sec. 4)
# ---------------------------------------------------------------------------
def schedule_conv2d(mod: tvm.IRModule) -> tvm.IRModule:
    """Find the conv2d TensorIR prim_func in ``mod`` and apply the course
    transformations (inline padding if present, fuse spatial loops, parallel +
    vectorize).  Returns the transformed IRModule.

    The conv2d prim_func generated by TOPI is named ``conv2d_nchw``.
    """
    sch = tvm.s_tir.Schedule(mod)
    conv_gv = None
    for gv, func in mod.functions_items():
        if "conv2d" in gv.name_hint:
            conv_gv = gv.name_hint
            break
    if conv_gv is None:
        raise RuntimeError("no conv2d prim_func found in module")

    block = sch.get_sblock("conv2d_nchw", func_name=conv_gv)
    # Inline the pad block if TOPI emitted one.
    try:
        pad_block = sch.get_sblock("pad_temp", func_name=conv_gv)
        sch.compute_inline(pad_block)
    except Exception:
        pass  # no padding block (valid conv)

    loops = sch.get_loops(block)
    # loops: n, co, oh, ow, [ci, kh, kw] -- parallelise the fused outer spatial
    # tile, keep the reduction loops innermost (untouched).
    n, co, oh, ow = loops[0], loops[1], loops[2], loops[3]
    fused = sch.fuse(n, co, oh)
    sch.parallel(fused)
    sch.vectorize(ow)
    return sch.mod


# ---------------------------------------------------------------------------
# NumPy reference for the whole network (matches PyTorch's forward)
# ---------------------------------------------------------------------------
def numpy_reference(x: np.ndarray, weight_map: dict) -> np.ndarray:
    """Vectorised NumPy forward pass used to check the Relax model."""
    import torch
    import torch.nn.functional as F

    xt = torch.from_numpy(x)
    w = {k: torch.from_numpy(v) for k, v in weight_map.items()}
    y = F.conv2d(xt, w["conv2d_weight"], w["conv2d_bias"])
    y = F.relu(y)
    y = F.max_pool2d(y, 2)
    y = torch.flatten(y, 1)
    y = F.linear(y, w["linear0_weight"], w["linear0_bias"])
    y = F.relu(y)
    y = F.linear(y, w["linear1_weight"], w["linear1_bias"])
    y = F.softmax(y, dim=1)
    return y.detach().numpy()
