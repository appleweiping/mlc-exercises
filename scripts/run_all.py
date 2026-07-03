"""Reproduce every measured result reported in the README and write evidence to
``results/``.

Produces:
* ``results/summary.json``     -- all measured numbers.
* ``results/run_all.log``      -- full text log.
* ``results/speedup.png``      -- schedule/tuning speedup bar chart.
* ``results/fashion_mnist_predictions.png`` -- sample compiled-model predictions.

Run:  python scripts/run_all.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("TVM_NUM_THREADS", "3")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
os.makedirs(RESULTS, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)

import ex1_tensorir as ex1  # noqa: E402
import ex2_end_to_end as ex2  # noqa: E402
import ex3_auto_optimize as ex3  # noqa: E402
import ex4_pytorch_integration as ex4  # noqa: E402
from mlc_compat import build_relax_vm, cpu, run_tir, time_tir, to_tvm  # noqa: E402

_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))


def main():
    summary = {}
    t_start = time.time()
    log("=" * 70)
    log("MLC exercises -- full verification run")
    import tvm

    log(f"apache-tvm {tvm.__version__} | numpy {np.__version__} | torch {torch.__version__}")
    log(f"threads: OMP={os.environ['OMP_NUM_THREADS']} TVM={os.environ['TVM_NUM_THREADS']}")
    log("=" * 70)

    # ------------------------------------------------------------------ EX1
    log("\n[Exercise 1] TensorIR matmul+ReLU (128x128)")
    a = np.random.rand(128, 128).astype("float32")
    b = np.random.rand(128, 128).astype("float32")
    ref = ex1.numpy_mm_relu(a, b)
    naive = run_tir(ex1.MatmulReLU, [a, b], (128, 128))
    sch = ex1.schedule_matmul_relu(ex1.MatmulReLU)
    tuned = run_tir(sch.mod, [a, b], (128, 128))
    t_naive = time_tir(ex1.MatmulReLU, [a, b], (128, 128), number=20, repeat=3)
    t_sched = time_tir(sch.mod, [a, b], (128, 128), number=20, repeat=3)
    log(f"  naive  err vs numpy: {np.max(np.abs(naive-ref)):.2e}  time: {t_naive*1e3:.3f} ms")
    log(f"  sched  err vs numpy: {np.max(np.abs(tuned-ref)):.2e}  time: {t_sched*1e3:.3f} ms")
    log(f"  hand-schedule speedup: {t_naive/t_sched:.2f}x")
    summary["ex1_matmul_relu"] = {
        "naive_err": float(np.max(np.abs(naive - ref))),
        "sched_err": float(np.max(np.abs(tuned - ref))),
        "naive_ms": t_naive * 1e3,
        "sched_ms": t_sched * 1e3,
        "speedup": t_naive / t_sched,
    }

    # ------------------------------------------------------------------ EX2
    log("\n[Exercise 2] End-to-end Fashion-MNIST model in Relax")
    params_path = os.path.join(DATA, "fasionmnist_mlp_assignment_params.pkl")
    with open(params_path, "rb") as f:  # official assignment weights (trusted)
        wm = pickle.load(f)
    x0 = np.random.rand(*ex2.INPUT_SHAPE).astype("float32")
    mod_te = ex2.create_model_via_emit_te(wm)
    out_te = _vm_run(mod_te, x0)
    ref_net = ex2.numpy_reference(x0, wm)
    err_te = float(np.max(np.abs(out_te - ref_net)))
    mod_torch = ex2.create_model_with_torch_func(wm)
    out_torch = _vm_run(mod_torch, x0)
    err_torch = float(np.max(np.abs(out_torch - ref_net)))
    log(f"  emit_te model    err vs torch: {err_te:.2e}")
    log(f"  vendor-func model err vs torch: {err_torch:.2e}")
    summary["ex2_model_err_emit_te"] = err_te
    summary["ex2_model_err_vendor_func"] = err_torch

    # Real Fashion-MNIST accuracy of the compiled Relax model.
    acc, n = _fashion_mnist_accuracy(mod_te)
    log(f"  compiled Relax model Fashion-MNIST test accuracy: {acc*100:.2f}% on {n} images")
    summary["ex2_fashion_mnist_accuracy"] = acc
    summary["ex2_fashion_mnist_n"] = n

    # ------------------------------------------------------------------ EX3
    log("\n[Exercise 3] Automatic program optimization (matmul 512x512)")
    aa = np.random.rand(ex3.N, ex3.N).astype("float32")
    bb = np.random.rand(ex3.N, ex3.N).astype("float32")
    t_base = time_tir(ex3.MatmulModule, [aa, bb], (ex3.N, ex3.N), number=3, repeat=3)
    best_sch, best_t, all_t = ex3.random_search(ex3.MatmulModule, num_trials=8, seed=3)
    log(f"  naive matmul time: {t_base*1e3:.2f} ms")
    log(f"  random-search best: {best_t*1e3:.2f} ms  ->  {t_base/best_t:.2f}x speedup")
    summary["ex3_naive_ms"] = t_base * 1e3
    summary["ex3_random_search_ms"] = best_t * 1e3
    summary["ex3_random_search_speedup"] = t_base / best_t

    ms_speedup = None
    try:
        import tempfile

        wd = tempfile.mkdtemp(prefix="ms_run_")
        ms_sch = ex3.tune_with_meta_schedule(
            ex3.MatmulModule, wd, max_trials=16, inputs=[aa, bb], out_shape=(ex3.N, ex3.N)
        )
        t_ms = time_tir(ms_sch.mod, [aa, bb], (ex3.N, ex3.N), number=3, repeat=3)
        ms_speedup = t_base / t_ms
        log(f"  meta-schedule best: {t_ms*1e3:.2f} ms  ->  {ms_speedup:.2f}x speedup")
        summary["ex3_meta_schedule_ms"] = t_ms * 1e3
        summary["ex3_meta_schedule_speedup"] = ms_speedup
    except Exception as exc:  # keep the rest of the run alive
        log(f"  meta-schedule run skipped: {exc}")

    # ------------------------------------------------------------------ EX4
    log("\n[Exercise 4] PyTorch -> Relax import (SmallCNN)")
    model = ex4.SmallCNN().eval()
    xin = np.random.rand(2, 1, 28, 28).astype("float32")
    ref4 = ex4.torch_reference(model, xin)
    mod4 = ex4.import_torch_model(model, (2, 1, 28, 28))
    out4 = ex4.build_and_run(mod4, xin)
    err4 = float(np.max(np.abs(out4 - ref4)))
    log(f"  imported+compiled SmallCNN err vs PyTorch: {err4:.2e}")
    summary["ex4_import_err"] = err4

    # ------------------------------------------------------------------ figures
    _speedup_figure(summary, ms_speedup)
    _prediction_figure(mod_te)

    summary["total_seconds"] = time.time() - t_start
    log(f"\nTotal wall time: {summary['total_seconds']:.1f} s")

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS, "run_all.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")
    log(f"\nWrote results/summary.json, results/run_all.log, and figures to {RESULTS}")


def _vm_run(mod, x):
    vm = build_relax_vm(mod)
    return vm["main"](to_tvm(x, cpu())).numpy()


def _fashion_mnist_accuracy(mod, max_images=2000):
    """Run the compiled Relax model over the real Fashion-MNIST test set."""
    try:
        import torchvision
        from torchvision import transforms

        ds = torchvision.datasets.FashionMNIST(
            root=DATA, train=False, download=True, transform=transforms.ToTensor()
        )
    except Exception as exc:
        log(f"  (Fashion-MNIST unavailable: {exc})")
        return 0.0, 0

    vm = build_relax_vm(mod)
    loader = torch.utils.data.DataLoader(ds, batch_size=ex2.BATCH, shuffle=False)
    correct, total = 0, 0
    for data, label in loader:
        if data.shape[0] != ex2.BATCH:
            continue
        out = vm["main"](to_tvm(data.numpy().astype("float32"), cpu())).numpy()
        pred = out.argmax(axis=1)
        correct += int((pred == label.numpy()).sum())
        total += ex2.BATCH
        if total >= max_images:
            break
    return correct / total, total


def _speedup_figure(summary, ms_speedup):
    labels = ["ex1 hand\nschedule", "ex3 random\nsearch"]
    vals = [summary["ex1_matmul_relu"]["speedup"], summary["ex3_random_search_speedup"]]
    if ms_speedup is not None:
        labels.append("ex3 meta\nschedule")
        vals.append(ms_speedup)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52"][: len(vals)])
    plt.ylabel("speedup vs naive (x)")
    plt.title("TVM schedule / auto-tuning speedups (CPU, 3 threads)")
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}x", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "speedup.png"), dpi=120)
    plt.close()


def _prediction_figure(mod, n=8):
    try:
        import torchvision
        from torchvision import transforms

        ds = torchvision.datasets.FashionMNIST(
            root=DATA, train=False, download=True, transform=transforms.ToTensor()
        )
    except Exception:
        return
    classes = [
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    vm = build_relax_vm(mod)
    plt.figure(figsize=(12, 2.2))
    for i in range(n):
        img, label = ds[i]
        batch = np.zeros(ex2.INPUT_SHAPE, dtype="float32")
        batch[0] = img.numpy()
        out = vm["main"](to_tvm(batch, cpu())).numpy()
        pred = int(out[0].argmax())
        ax = plt.subplot(1, n, i + 1)
        ax.imshow(img.numpy()[0], cmap="gray")
        ax.set_title(f"{classes[pred]}\n(gt {classes[label]})", fontsize=8,
                     color="green" if pred == label else "red")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fashion_mnist_predictions.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
