"""Small compatibility / helper layer for the MLC exercises.

The MLC (mlc.ai) course was written against the ``mlc-ai-nightly`` *TVM Unity*
wheels, whose Python API used ``tvm.tir.Schedule``, ``tvm.script.tir as T`` and
``tvm.nd``.  Those wheels ship Linux/macOS only.  This repo runs on the public
``apache-tvm`` 0.25 wheel, which is the *post-Unity-refactor* build with a
renamed API surface:

===============================  ================================================
course-era (mlc-ai-nightly)      apache-tvm 0.25 (this repo)
===============================  ================================================
``from tvm.script import tir``   ``from tvm.script import tirx as T`` (+ ``s_tir=True``)
``@tvm.script.ir_module``        ``@ir_module(s_tir=True)``
``@T.prim_func``                 ``@T.prim_func(s_tir=True)``
``with T.block(...)``            ``with T.sblock(...)``
``tvm.tir.Schedule``             ``tvm.s_tir.Schedule``
``sch.get_block(...)``           ``sch.get_sblock(...)``
``tvm.build`` / ``relax.build``  ``tvm.compile(...).jit()``
``tvm.nd.array`` / ``tvm.cpu``   ``tvm.runtime.tensor`` / ``tvm.runtime.cpu``
``relax.DynTensorType``          ``relax.TensorStructInfo``
``tvm.meta_schedule``            ``tvm.s_tir.meta_schedule``
===============================  ================================================

Keeping these helpers in one place means every exercise module stays readable
and close to the original course notebooks.
"""

from __future__ import annotations

import numpy as np
import tvm

CPU_TARGET = "llvm"


def cpu():
    """Return the CPU device."""
    return tvm.runtime.cpu()


def to_tvm(arr: np.ndarray, dev=None):
    """NumPy -> TVM tensor on ``dev`` (CPU by default)."""
    return tvm.runtime.tensor(np.ascontiguousarray(arr), dev or cpu())


def empty(shape, dtype="float32", dev=None):
    """Allocate an uninitialised TVM tensor."""
    return tvm.runtime.empty(shape, dtype, dev or cpu())


def build_tir(mod, target: str = CPU_TARGET):
    """Compile a TensorIR IRModule and return a callable runtime module."""
    return tvm.compile(mod, target=target).jit()


def run_tir(mod, inputs, out_shape, out_dtype="float32", func_name="main", target=CPU_TARGET):
    """Build ``mod``, run ``func_name`` with the given NumPy ``inputs`` and return
    the output as a NumPy array.  ``inputs`` are NumPy arrays; a single output
    buffer of shape ``out_shape`` is appended (DPS calling convention)."""
    rt = build_tir(mod, target)
    dev = cpu()
    args = [to_tvm(a, dev) for a in inputs]
    out = empty(out_shape, out_dtype, dev)
    rt[func_name](*args, out)
    return out.numpy()


def build_relax_vm(mod, target: str = CPU_TARGET):
    """Compile a Relax IRModule and wrap it in a VirtualMachine."""
    from tvm import relax

    ex = tvm.compile(mod, target=target)
    return relax.VirtualMachine(ex, cpu())


def time_tir(mod, inputs, out_shape, out_dtype="float32", func_name="main",
             number=10, repeat=3, target=CPU_TARGET):
    """Return median wall-clock seconds per run of ``func_name`` using TVM's
    own ``time_evaluator`` (warm, excludes build)."""
    rt = build_tir(mod, target)
    dev = cpu()
    args = [to_tvm(a, dev) for a in inputs]
    out = empty(out_shape, out_dtype, dev)
    timer = rt.time_evaluator(func_name, dev, number=number, repeat=repeat)
    res = timer(*args, out)
    return float(np.median(res.results))
