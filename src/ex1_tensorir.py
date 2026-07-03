"""MLC Exercise 1 -- TensorIR: Tensor Program Abstraction & Transformations.

Grounded on the MLC (mlc.ai, Tianqi Chen) lecture notebooks
``2_tensor_program_abstraction`` and
``3_TensorIR_Tensor_Program_Abstraction_Case_Study_Action``.

Implemented from scratch with real TVM (apache-tvm 0.25) APIs:

1. ``MatmulReLU`` -- a fused (128,128) matmul + ReLU written directly in
   TVMScript (block-based TensorIR), built and checked against a NumPy reference.
2. ``te_matmul_relu`` -- the same program constructed from a Tensor Expression
   description and lowered with ``te.create_prim_func``.
3. ``schedule_matmul_relu`` -- the canonical loop transformations taught in the
   course (``split`` / ``reorder`` / ``decompose_reduction`` / ``parallel`` /
   ``vectorize`` / ``unroll``) applied through ``tvm.s_tir.Schedule`` and verified
   to preserve numerics while restructuring the loop nest.
4. ``te_bmm_bias_relu`` + ``schedule_bmm`` -- a batched matmul + broadcast
   bias-add + ReLU "case study" program, transformed and re-verified.

All programs target CPU (``llvm``) and are exercised by
``tests/test_ex1_tensorir.py``.
"""

from __future__ import annotations

import numpy as np
import tvm
from tvm import te
from tvm.script import ir_module
from tvm.script import tirx as T


# ---------------------------------------------------------------------------
# 1. A TensorIR program written directly in TVMScript: C = relu(A @ B)
# ---------------------------------------------------------------------------
@ir_module(s_tir=True)
class MatmulReLU:
    """Fused (128,128) x (128,128) matmul followed by ReLU (MLC lecture 2)."""

    @T.prim_func(s_tir=True)
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), "float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.sblock("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


def numpy_mm_relu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference the TensorIR program must match."""
    return np.maximum(a @ b, 0.0)


# ---------------------------------------------------------------------------
# 2. Build the same program from a Tensor Expression (TE) description
# ---------------------------------------------------------------------------
def te_matmul_relu(n: int = 128) -> tvm.IRModule:
    """Describe matmul + ReLU with Tensor Expression, lower to an IRModule."""
    A = te.placeholder((n, n), name="A", dtype="float32")
    B = te.placeholder((n, n), name="B", dtype="float32")
    k = te.reduce_axis((0, n), name="k")
    Y = te.compute((n, n), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
    C = te.compute((n, n), lambda i, j: te.max(Y[i, j], 0.0), name="C")
    func = te.create_prim_func([A, B, C])
    return tvm.IRModule({"main": func})


# ---------------------------------------------------------------------------
# 3. Course-taught schedule transformations on the matmul block
# ---------------------------------------------------------------------------
def schedule_matmul_relu(mod: tvm.IRModule) -> "tvm.s_tir.Schedule":
    """Tile the output, keep the reduction innermost, and map loops to hardware
    axes using the primitives taught in MLC lecture 2/3:
    ``split`` / ``reorder`` / ``fuse`` / ``parallel`` / ``unroll`` / ``vectorize``.

    This is NOT a unique "answer" -- it is one correct schedule.  Correctness is
    verified by rebuilding and comparing against the NumPy reference.

    Note on legality: ``parallel``/``vectorize`` may only touch loops that do not
    straddle a reduction's ``init``.  We therefore parallelise the *tiled spatial*
    loops of the matmul block (reduction ``k`` stays innermost) and vectorise the
    reduction-free ReLU epilogue after moving it out with ``fuse`` + ``split``.
    """
    sch = tvm.s_tir.Schedule(mod)
    block_Y = sch.get_sblock("Y", func_name="main")
    i, j, k = sch.get_loops(block_Y)

    # Tile the two spatial loops (8-wide inner factor, as in the course).
    i0, i1 = sch.split(i, factors=[None, 8])
    j0, j1 = sch.split(j, factors=[None, 8])
    # Reorder so the reduction sits innermost.
    sch.reorder(i0, j0, i1, j1, k)
    # Fuse the outer tile loops and parallelise across output tiles.
    io = sch.fuse(i0, j0)
    sch.parallel(io)
    sch.unroll(i1)

    # ReLU epilogue (pure spatial) -> parallel + vectorized.
    block_C = sch.get_sblock("C", func_name="main")
    ci, cj = sch.get_loops(block_C)
    cio = sch.fuse(ci, cj)
    c0, c1 = sch.split(cio, factors=[None, 8])
    sch.parallel(c0)
    sch.vectorize(c1)
    return sch


# ---------------------------------------------------------------------------
# 4. Batched matmul + broadcast bias-add + ReLU (the "case study" program)
# ---------------------------------------------------------------------------
def te_bmm_bias_relu(batch: int = 4, n: int = 64, m: int = 64, k: int = 64) -> tvm.IRModule:
    """batched (batch,n,k) @ (batch,k,m) + bias(m), then ReLU."""
    X = te.placeholder((batch, n, k), name="X", dtype="float32")
    W = te.placeholder((batch, k, m), name="W", dtype="float32")
    bias = te.placeholder((m,), name="bias", dtype="float32")
    rk = te.reduce_axis((0, k), name="rk")
    Y = te.compute(
        (batch, n, m),
        lambda b, i, j: te.sum(X[b, i, rk] * W[b, rk, j], axis=rk),
        name="Y",
    )
    Z = te.compute(
        (batch, n, m), lambda b, i, j: te.max(Y[b, i, j] + bias[j], 0.0), name="Z"
    )
    func = te.create_prim_func([X, W, bias, Z])
    return tvm.IRModule({"main": func})


def numpy_bmm_bias_relu(x, w, bias):
    return np.maximum(np.matmul(x, w) + bias.reshape(1, 1, -1), 0.0)


def schedule_bmm(mod: tvm.IRModule) -> "tvm.s_tir.Schedule":
    """Parallelise across the batch axis and vectorise the inner spatial loop."""
    sch = tvm.s_tir.Schedule(mod)
    blk = sch.get_sblock("Y", func_name="main")
    b, i, j, k = sch.get_loops(blk)
    sch.reorder(b, i, k, j)
    sch.parallel(b)
    sch.vectorize(j)
    return sch


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from mlc_compat import run_tir

    a = np.random.rand(128, 128).astype("float32")
    b = np.random.rand(128, 128).astype("float32")
    sch = schedule_matmul_relu(MatmulReLU)
    out = run_tir(sch.mod, [a, b], (128, 128))
    print("matmul+relu max err:", float(np.max(np.abs(out - numpy_mm_relu(a, b)))))
