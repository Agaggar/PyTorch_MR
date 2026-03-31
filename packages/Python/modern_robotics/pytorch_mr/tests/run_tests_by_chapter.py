#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class ChapterRun:
    name: str
    path: str
    collected: int
    exit_code: int


def _collect_count(pytest_args, env):
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *pytest_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    # Each collected test item prints one line like: path::test_name
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    items = [ln for ln in lines if "::" in ln and not ln.startswith("<") and "warnings summary" not in ln.lower()]
    return len(items)


def _run(pytest_args, env):
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *pytest_args],
        env=env,
    )
    return p.returncode


def main():
    here = os.path.abspath(os.path.dirname(__file__))
    # packages/Python (so `modern_robotics/...` paths exist)
    pkg_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    chapters = [
        ("ch1_3", os.path.join(here, "test_ch1_3_numpy_parity.py")),
        ("ch3", os.path.join(here, "test_ch3.py")),
        ("ch4_6", os.path.join(here, "test_ch4_6.py")),
        ("ch8", os.path.join(here, "test_ch8.py")),
        ("ch9_11", os.path.join(here, "test_ch9_11.py")),
    ]

    runs = []
    total_collected = 0
    any_failed = False

    for name, path in chapters:
        rel = os.path.relpath(path, pkg_root)
        collected = _collect_count([rel], env)
        exit_code = _run([rel], env)
        runs.append(ChapterRun(name=name, path=rel, collected=collected, exit_code=exit_code))
        total_collected += collected
        any_failed = any_failed or (exit_code != 0)

    print("\n=== Chapter test summary ===")
    for r in runs:
        status = "PASS" if r.exit_code == 0 else "FAIL"
        print(f"{r.name:>6} | {status} | collected={r.collected:4d} | {r.path}")
    print(f"TOTAL | {'PASS' if not any_failed else 'FAIL'} | collected={total_collected:4d}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

