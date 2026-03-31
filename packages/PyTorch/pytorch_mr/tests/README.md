# Unit tests for `pytorch_mr`

Tests live next to the PyTorch package under `packages/PyTorch/pytorch_mr/tests/`.

They compare outputs to the NumPy reference `modern_robotics` (`packages/Python`). Ensure both are importable:

- From repo `packages/` directory:

```bash
PYTHONPATH=PyTorch:Python python -m pytest PyTorch/pytorch_mr/tests -q
```

- Chapter summary:

```bash
PYTHONPATH=PyTorch:Python python PyTorch/pytorch_mr/tests/run_tests_by_chapter.py
```

After `pip install -e .` from `packages/PyTorch`, `pip install pytest`, and a NumPy reference (`pip install modern_robotics` or `pip install -e ../Python`), run:

```bash
PYTHONPATH=/path/to/repo/packages/Python python -m pytest --pyargs pytorch_mr.tests -q
```

(Use `PYTHONPATH` to point at the local `modern_robotics` tree when you are not using the PyPI package.)
