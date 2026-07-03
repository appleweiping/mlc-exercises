"""Download the datasets/weights the MLC assignment 1 uses (kept out of git).

- ``fasionmnist_mlp_assignment_params.pkl`` : the assignment's own pre-trained
  weight map for the Fashion-MNIST classifier (~83% test accuracy).
- Fashion-MNIST test set via torchvision (for the real end-to-end accuracy run).

Everything lands under ``data/`` which is git-ignored.
"""

from __future__ import annotations

import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

PARAMS_URL = (
    "https://github.com/mlc-ai/web-data/raw/main/models/"
    "fasionmnist_mlp_assignment_params.pkl"
)
PARAMS_PATH = os.path.join(DATA, "fasionmnist_mlp_assignment_params.pkl")


def download_params(path: str = PARAMS_PATH) -> str:
    if not os.path.exists(path):
        print(f"downloading {PARAMS_URL}")
        urllib.request.urlretrieve(PARAMS_URL, path)
    print("params at", path, os.path.getsize(path), "bytes")
    return path


def download_fashion_mnist():
    import torchvision
    from torchvision import transforms

    ds = torchvision.datasets.FashionMNIST(
        root=DATA, train=False, download=True, transform=transforms.ToTensor()
    )
    print("Fashion-MNIST test set:", len(ds), "samples")
    return ds


if __name__ == "__main__":
    download_params()
    try:
        download_fashion_mnist()
    except Exception as exc:  # torchvision download can be flaky behind proxies
        print("Fashion-MNIST download skipped:", exc)
