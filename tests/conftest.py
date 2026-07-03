"""Make ``src/`` importable and enforce CPU-only, low-thread execution."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("TVM_NUM_THREADS", "3")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import torch

    torch.set_num_threads(3)
except Exception:  # torch is only needed by some tests
    pass
