import os
import sys
from typing import Any, Callable, Sequence

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(__file__)

# Prefer installed packages; fall back to sibling repo layout (packages/PyTorch + packages/Python).
try:
    import modern_robotics.core as mr_np  # noqa: E402
except ImportError:
    _PKG_PYTHON_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "Python"))
    if _PKG_PYTHON_ROOT not in sys.path:
        sys.path.insert(0, _PKG_PYTHON_ROOT)
    import modern_robotics.core as mr_np  # noqa: E402

try:
    import pytorch_mr.core as mr_t  # noqa: E402
except ImportError:
    _PKG_TORCH_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
    if _PKG_TORCH_ROOT not in sys.path:
        sys.path.insert(0, _PKG_TORCH_ROOT)
    import pytorch_mr.core as mr_t  # noqa: E402


@pytest.fixture(scope="session")
def np_ref():
    return mr_np


@pytest.fixture(scope="session")
def pt_impl():
    return mr_t


def to_numpy(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def to_tensor(x: Any, *, dtype=None, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


def _iter_batch(x: torch.Tensor) -> Sequence[torch.Tensor]:
    if x.ndim == 0:
        return [x]
    if x.ndim == 1:
        return [x]
    if x.ndim >= 2:
        return [x[i] for i in range(x.shape[0])]
    return [x]


def assert_torch_matches_numpy(
    torch_out: Any,
    numpy_out: Any,
    *,
    atol=1e-6,
    rtol=1e-6,
):
    if isinstance(torch_out, tuple):
        assert isinstance(numpy_out, tuple)
        assert len(torch_out) == len(numpy_out)
        for a, b in zip(torch_out, numpy_out):
            assert_torch_matches_numpy(a, b, atol=atol, rtol=rtol)
        return

    if isinstance(torch_out, torch.Tensor):
        np_t = torch_out.detach().cpu().numpy()
        np_n = np.array(numpy_out)
        assert np_t.shape == np_n.shape
        assert np.allclose(np_t, np_n, atol=atol, rtol=rtol), (np_t, np_n)
        return

    assert torch_out == numpy_out


def apply_numpy_per_batch(
    f_np: Callable[..., Any],
    *args_t: Any,
    squeeze_last_dim1: bool = True,
) -> Any:
    """
    Call NumPy reference function `f_np` on each batch element.

    For tensor args:
    - If shape is (N, ..., 1) and squeeze_last_dim1=True, squeeze the last dim
      before passing to NumPy (to align with reference signatures).
    - If shape is (N, ...), iterate over axis 0.
    - If shape is (...), treat as single item.
    """
    batch = None
    for a in args_t:
        if isinstance(a, torch.Tensor) and a.ndim >= 2:
            batch = a.shape[0]
            break
    if batch is None:
        np_args = []
        for a in args_t:
            if isinstance(a, torch.Tensor):
                x = a
                if squeeze_last_dim1 and x.ndim >= 1 and x.shape[-1] == 1:
                    x = x.squeeze(-1)
                np_args.append(to_numpy(x))
            else:
                np_args.append(a)
        return f_np(*np_args)

    outs = []
    for i in range(batch):
        np_args_i = []
        for a in args_t:
            if isinstance(a, torch.Tensor):
                x = a[i] if a.ndim >= 2 else a
                if squeeze_last_dim1 and x.ndim >= 1 and x.shape[-1] == 1:
                    x = x.squeeze(-1)
                np_args_i.append(to_numpy(x))
            else:
                np_args_i.append(a)
        outs.append(f_np(*np_args_i))
    return outs
