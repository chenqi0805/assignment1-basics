---
name: local-dev
description: How to set up and verify local dev for this CS336 assignment repo (uv + pytest)
---

# local-dev — chenqi0805/assignment1-basics

Durable record of the onboarding LOCAL-DEV run (2026-10-09). Future workers: follow
these exact steps; they are known-good in the snapshot.

## Steps that worked

1. **Install uv** (not preinstalled): `curl -LsSf https://astral.sh/uv/install.sh | sh`
   → installs to `~/.local/bin` (uv 0.12.24). New shells need `export PATH="$HOME/.local/bin:$PATH"`.
2. **Sync env**: `uv sync` from repo root. Downloads a uv-managed Python (3.13.14 here)
   plus torch 2.6.0+cu124 and CUDA wheels (~2 GB, a few minutes). `uv.lock` is committed —
   sync is deterministic. Exits 0.
3. **Verify imports**: `uv run python -c "import torch, cs336_basics; print(torch.__version__, cs336_basics.__version__)"`.
4. **Canonical dev command**: `uv run pytest` (README). Full suite = 8 test files, 48 tests,
   ~10 s CPU. Healthy fresh checkout = 47 FAILED (all `NotImplementedError`) + 1 xfailed,
   0 collection/import errors. This is expected — see "Why failures are healthy".
5. **Functional smoke**: the only provided implementation is `find_chunk_boundaries` in
   `cs336_basics/pretokenization_example.py`. It is NOT importable (module-level
   `open(...)` example code raises `TypeError` on import). Exec the source up to the
   `## Usage` marker and call the function against `tests/fixtures/tinystories_sample.txt`
   with `num_processes=4`, token `b"<|endoftext|>"` → `[0, 1412, 1940, 3780, 3794]`.
6. **Typecheck**: `uv run ty check cs336_basics` → "All checks passed!" (ignore the
   pre-release WARN banner).
7. **Lint**: `uvx ruff check cs336_basics tests` → 12 style nits in upstream starter code
   (4 I001, 3 F402, 1 W605, 1 R0402, 2 M115/M118, 1 F007). Informational; ruff config
   exists in pyproject.toml (line-length 120, extend-select UP) but ruff is not a dependency.

## Why failures are healthy

The graded suite calls `tests/adapters.py::run_*` functions, which all
`raise NotImplementedError` until the student implements `cs336_basics/`. The README
states: "Initially, all tests should fail with NotImplementedErrors." A healthy env is
proven by **collection success + uniform NotImplementedError failures**, not green tests.
Once the student's implementation lands, all 48 should pass (1 xfail stays expected:
snapshot comparison test).

## Gotchas

- No services, no env vars, no ports. Don't look for a server to start — there isn't one.
- `data/` downloads (TinyStories/OWT from HuggingFace) are only needed for the training
  experiments in the handout, never for the test suite.
- `make_submission.sh` swallows pytest failures (`|| true`) on purpose.
- Missing dependency → don't `pip install`; edit `pyproject.toml` and `uv sync`/`uv lock`.

## Validation Summary (onboarding run)

- uv sync: EXIT 0 — Python 3.13.14, torch 2.6.0+cu124, cs336_basics 1.0.5 importable.
- `uv run pytest -v`: 47 failed / 1 xfailed in 10.21 s; every failure `NotImplementedError`; 0 collection errors.
- Functional exercise of provided code: `find_chunk_boundaries` → `[0, 1412, 1940, 3780, 3794]` (asserts passed).
- `ty check cs336_basics`: all checks passed. `uvx ruff check`: 12 informational style nits.
- Verdict: **dev_stack_healthy = true** (starter-repo sense: env syncs, suite runs, failures are the documented initial state).
