# PyTorch Modern Robotics (`pytorch_mr`)

PyTorch implementation of the routines from *Modern Robotics: Mechanics, Planning, and Control*, aligned with the NumPy `modern_robotics` reference in `packages/Python`.

## Install (global)
```bash
pip install pytorch-mr
```

## Install (local / editable)

From this directory (`packages/PyTorch`):

```bash
pip install -e .
```

Dependencies are declared in `pyproject.toml` (`numpy`, `torch`).

## Import

```python
import pytorch_mr as pmr
# or
from pytorch_mr.core import MatrixExp3, FKinBody
```

## Tests (NumPy parity)

Install the reference `modern_robotics` (local tree or PyPI) and `pytest`:

```bash
pip install -e ../Python   # local reference
# or
pip install modern_robotics
pip install pytest
```

From `packages/PyTorch` (uses `[tool.pytest.ini_options]` `pythonpath = ["src"]`):

```bash
cd packages/PyTorch
python -m pytest tests -q
```

From `packages/` with explicit `PYTHONPATH`:

```bash
cd packages
PYTHONPATH=PyTorch/src:Python python -m pytest PyTorch/tests -q
```

Chapter summary:

```bash
cd packages
PYTHONPATH=PyTorch/src:Python python PyTorch/tests/run_tests_by_chapter.py
```
