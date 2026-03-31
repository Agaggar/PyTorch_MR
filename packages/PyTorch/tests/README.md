# Unit tests for `pytorch_mr`

Tests live under `packages/PyTorch/tests/` (sibling of `src/`), matching a standard PyPA layout.

They compare outputs to the NumPy reference `modern_robotics` in `packages/Python`. `conftest.py` adds `packages/Python` to `sys.path` when `modern_robotics` is not installed.

**Recommended (from `packages/PyTorch`):**

```bash
python -m pytest tests -q
```

**From `packages/`:**

```bash
PYTHONPATH=PyTorch/src:Python python -m pytest PyTorch/tests -q
```

**Chapter summary:**

```bash
cd packages
PYTHONPATH=PyTorch/src:Python python PyTorch/tests/run_tests_by_chapter.py
```

With `pip install modern_robotics` (or editable `../Python`), you only need `src` on the path for an uninstalled checkout; after `pip install -e .`, `pytorch_mr` resolves without extra `PYTHONPATH`.
