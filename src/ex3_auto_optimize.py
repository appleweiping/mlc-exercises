"""MLC Exercise 3 -- Automatic Program Optimization (meta-schedule tuning).

Grounded on the MLC lecture 5 notebook (*Automatic Program Optimization*).

Instead of hand-writing a schedule (Exercise 1), we let TVM's **meta-schedule**
search over a large space of loop tilings / vectorisations / parallelisations,
using a cost model to pick the fastest.  This module:

1. Defines a matmul TensorIR program (``matmul_module``).
2. ``random_search`` -- the "stochastic schedule" idea from the course: sample
   ``sample_perfect_tile`` factors and keep the best of N random schedules
   (a mini meta-schedule, self-contained, no XGBoost needed).
3. ``tune_with_meta_schedule`` -- run the real ``meta_schedule.tune_tir`` search
   and return the best tuned Schedule via ``compile_tir``.

All results are numerically verified against NumPy, and wall-clock timings are
measured with TVM's ``time_evaluator`` so the reported speedups are real.
"""

from __future__ import annotations

import numpy as np
import tvm
from tvm.script import ir_module
from tvm.script import tirx as T

N = 512  # matmul size for tuning (CPU-scale but non-trivial)


@ir_module(s_tir=True)
class MatmulModule:
    @T.prim_func(s_tir=True)
    def main(
        A: T.Buffer((512, 512), "float32"),
        B: T.Buffer((512, 512), "float32"),
        C: T.Buffer((512, 512), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(512, 512, 512):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def numpy_matmul(a, b):
    return a @ b


# ---------------------------------------------------------------------------
# 2. Stochastic schedule search (the "random search" from lecture 5)
# ---------------------------------------------------------------------------
def stochastic_schedule(sch: "tvm.s_tir.Schedule") -> "tvm.s_tir.Schedule":
    """One randomly-tiled schedule using ``sample_perfect_tile`` (course idiom)."""
    block = sch.get_sblock("C", func_name="main")
    i, j, k = sch.get_loops(block)
    i_factors = sch.sample_perfect_tile(i, n=2)
    j_factors = sch.sample_perfect_tile(j, n=2)
    i0, i1 = sch.split(i, factors=i_factors)
    j0, j1 = sch.split(j, factors=j_factors)
    sch.reorder(i0, j0, k, i1, j1)
    sch.parallel(sch.fuse(i0, j0))
    sch.vectorize(j1)
    return sch


def random_search(mod: tvm.IRModule, num_trials: int = 8, seed: int = 0):
    """Keep the fastest of ``num_trials`` random schedules.  Returns
    ``(best_schedule, best_seconds, all_seconds)``.
    """
    from mlc_compat import time_tir

    rng = np.random.default_rng(seed)
    a = rng.random((N, N), dtype=np.float32)
    b = rng.random((N, N), dtype=np.float32)
    ref = numpy_matmul(a, b)

    best_sch, best_t, times = None, float("inf"), []
    for t in range(num_trials):
        sch = tvm.s_tir.Schedule(mod, seed=int(rng.integers(1 << 30)))
        stochastic_schedule(sch)
        # correctness gate
        from mlc_compat import run_tir

        out = run_tir(sch.mod, [a, b], (N, N))
        assert np.allclose(out, ref, rtol=1e-2, atol=1e-2), "random schedule wrong"
        sec = time_tir(sch.mod, [a, b], (N, N), number=3, repeat=2)
        times.append(sec)
        if sec < best_t:
            best_t, best_sch = sec, sch
    return best_sch, best_t, times


# ---------------------------------------------------------------------------
# 3. Real meta-schedule tuning
# ---------------------------------------------------------------------------
# apache-tvm 0.25 no longer accepts CLI target strings ("llvm --num-cores=3");
# targets must be given as a JSON/dict spec.
CPU_TARGET = {"kind": "llvm", "num-cores": 3}


def tune_with_meta_schedule(mod: tvm.IRModule, work_dir: str, max_trials: int = 32,
                            target=None, inputs=None, out_shape=None):
    """Run the real ``meta_schedule.tune_tir`` evolutionary search and return the
    best tuned Schedule.

    On native Windows the meta-schedule *local runner* (a ``PopenPoolExecutor``
    subprocess builder/runner) cannot execute the measurement subprocess, so the
    JSON database ends up with sentinel ``run_secs = 1e10`` for every trial.  The
    *search* itself -- space generation, evolutionary strategy, cost model and
    database -- all run correctly; only the on-device timing is missing.

    To still produce a genuinely-best schedule we replay each candidate trace the
    search produced and **time it in-process** (which works fine on Windows),
    returning the fastest verified candidate.  If ``inputs`` are not supplied we
    fall back to meta-schedule's own ``compile_tir`` selection.
    """
    from mlc_compat import run_tir, time_tir

    ms = tvm.s_tir.meta_schedule
    tgt = tvm.target.Target(target or CPU_TARGET)
    database = ms.tune_tir(
        mod=mod,
        target=tgt,
        work_dir=work_dir,
        max_trials_global=max_trials,
        num_trials_per_iter=16,
        seed=0,
    )

    records = list(database.get_all_tuning_records())
    device_timed = [r for r in records if float(r.run_secs[0]) < 1e9]
    if device_timed:  # on Linux/macOS the runner works -> trust the database
        return ms.tir_integration.compile_tir(database, mod, tgt)

    if inputs is None or out_shape is None:
        # No inputs to re-time with; return meta-schedule's structural choice.
        return ms.tir_integration.compile_tir(database, mod, tgt)

    # Windows path: replay each candidate trace and time it in-process.
    best_sch, best_t = None, float("inf")
    for rec in records:
        sch = tvm.s_tir.Schedule(mod)
        try:
            rec.trace.apply_to_schedule(sch, remove_postproc=False)
            out = run_tir(sch.mod, inputs, out_shape)
        except Exception:
            continue
        ref = numpy_matmul(*inputs)
        if not np.allclose(out, ref, rtol=1e-2, atol=1e-2):
            continue
        sec = time_tir(sch.mod, inputs, out_shape, number=3, repeat=2)
        if sec < best_t:
            best_t, best_sch = sec, sch
    if best_sch is None:  # extreme fallback: default lowering
        return tvm.s_tir.Schedule(mod)
    return best_sch
