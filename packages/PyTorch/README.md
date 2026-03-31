# PyTorch Modern Robotics (`pytorch_mr`)

PyTorch implementation of the routines from *Modern Robotics: Mechanics, Planning, and Control*, aligned with the NumPy `modern_robotics` reference in `packages/Python`.

## Install

From this directory (`packages/PyTorch`):

```bash
pip install -e .
```

Dependencies: `numpy`, `torch` (see `setup.py`).

## Import

```python
import pytorch_mr as pmr
# or
from pytorch_mr.core import MatrixExp3, FKinBody
```

## Tests (NumPy parity)

Tests compare against the reference `modern_robotics` package. Install the NumPy library from the sibling directory or from PyPI:

```bash
pip install -e ../Python   # local reference, from packages/PyTorch
# or
pip install modern_robotics
pip install pytest
```

Run tests with both package roots on `PYTHONPATH` (from `packages/` in this repo):

```bash
cd packages
PYTHONPATH=PyTorch:Python python -m pytest PyTorch/pytorch_mr/tests -q
```

Or use the chapter summary script:

```bash
cd packages
PYTHONPATH=PyTorch:Python python PyTorch/pytorch_mr/tests/run_tests_by_chapter.py
```

## Publishing to PyPI (optional)

Build and upload with standard `twine` / `build` workflows; ensure the package name `pytorch_mr` is available on your target index.
