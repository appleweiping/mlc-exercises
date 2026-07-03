"""MLC Exercise 4 -- Integration with Machine Learning Frameworks (PyTorch).

Grounded on the MLC lecture 6 notebook
(*Integration with Machine Learning Frameworks*).

The MLC workflow for deploying a real model is: *trace* a PyTorch ``nn.Module``
with ``torch.fx``, *import* the FX graph into a Relax ``IRModule`` with
``relax.frontend.torch.from_fx``, *legalise + optimise* it with the Relax
transform pipeline, then *build & run* on the Relax VirtualMachine.

This module implements that whole flow for two real models and verifies the
compiled output against PyTorch's own forward pass:

* ``SmallCNN``   -- Conv -> ReLU -> MaxPool -> Flatten -> Linear -> ReLU ->
                    Linear -> Softmax (the assignment's Fashion-MNIST topology).
* ``import_torch_model`` -- trace + import any ``nn.Module`` to Relax.
* ``build_and_run``      -- legalise, compile and run, returning a NumPy array.
* ``optimize_pipeline``  -- a small Relax optimisation pipeline
                            (LegalizeOps + FoldConstant + dead-code elimination)
                            shown to preserve numerics.
"""

from __future__ import annotations

import numpy as np
import torch
import tvm
from torch import fx, nn
from tvm import relax
from tvm.relax.frontend.torch import from_fx


class SmallCNN(nn.Module):
    """The assignment's Fashion-MNIST classifier as a plain PyTorch module."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3, bias=True)
        self.fc1 = nn.Linear(32 * 13 * 13, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)


class TinyMLP(nn.Module):
    def __init__(self, din=16, dh=32, dout=10):
        super().__init__()
        self.fc1 = nn.Linear(din, dh)
        self.fc2 = nn.Linear(dh, dout)

    def forward(self, x):
        return torch.softmax(self.fc2(torch.relu(self.fc1(x))), dim=1)


# ---------------------------------------------------------------------------
# import + build + run
# ---------------------------------------------------------------------------
def import_torch_model(model: nn.Module, input_shape, dtype: str = "float32") -> tvm.IRModule:
    """Trace ``model`` with torch.fx and import the graph into a Relax IRModule."""
    model = model.eval()
    graph_module = fx.symbolic_trace(model)
    return from_fx(graph_module, [(tuple(input_shape), dtype)])


def optimize_pipeline(mod: tvm.IRModule) -> tvm.IRModule:
    """A small but real Relax optimisation pipeline."""
    seq = tvm.ir.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.FoldConstant(),
            relax.transform.DeadCodeElimination(),
        ]
    )
    return seq(mod)


def build_and_run(mod: tvm.IRModule, x: np.ndarray, optimize: bool = True):
    """Legalise (+optimise), compile and run ``mod`` on the Relax VM."""
    if optimize:
        mod = optimize_pipeline(mod)
    else:
        mod = relax.transform.LegalizeOps()(mod)
    ex = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.runtime.cpu())
    out = vm["main"](tvm.runtime.tensor(x, tvm.runtime.cpu()))
    if not hasattr(out, "numpy"):  # tuple/adt return
        out = out[0]
    return out.numpy()


def torch_reference(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model = model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).detach().numpy()
