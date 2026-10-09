# obvious.md — chenqi0805/assignment1-basics

CS336 Spring 2025 Assignment 1 ("basics"): build an LLM from scratch — BPE tokenizer,
Transformer (linear, embedding, RMSNorm, RoPE, SwiGLU, attention), AdamW optimizer,
and a training loop. **This is a course-work starter repo**: the student implements
`cs336_basics/` and connects it to the graded test suite via `tests/adapters.py`.

## Stack

| Component | Value |
|---|---|
| Language | Python >= 3.11 (uv-managed interpreter; sandbox synced with 3.13.14) |
| Package manager | `uv` (uv.lock committed; uv_build backend; install: `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| Core deps | torch ~= 2.6.0, numpy, einops, einx, jaxtyping, regex, tiktoken, tqdm, wandb |
| Test framework | pytest (config in `pyproject.toml`: `log_cli`, `-s` addopts) |
| Lint / typecheck | ruff (config only, run via `uvx ruff`); `ty` (pre-release, in deps) |

## Commands

```sh
uv sync                                # create/sync .venv from uv.lock (downloads uv-managed Python + torch)
uv run pytest                          # full test suite (canonical dev command)
uv run pytest tests/test_model.py -v   # one test file
uv run <python_file_path>              # run any script in the auto-managed env
uv run ty check cs336_basics           # typecheck (ty is pre-release; warnings about that are normal)
uvx ruff check cs336_basics tests      # lint (ruff configured in pyproject.toml, line-length 120)
bash make_submission.sh                # run tests + zip submission artifact
```

No ports, no servers, no background services, no required env vars, no Docker/Compose/Makefile.

## Critical expectations (do not "fix" these)

- **A fresh checkout fails `uv run pytest` with 47 `NotImplementedError`s — that is the
  documented healthy initial state** (README: "Initially, all tests should fail with
  NotImplementedErrors"). Tests pass only as the student implements modules and wires
  them in `tests/adapters.py`.
- Environment health = `uv sync` exits 0, pytest **collects and runs** all 8 test files
  with zero collection/import errors, and every failure is `NotImplementedError`.
- `cs336_basics/pretokenization_example.py` is **not importable** — it executes example
  usage (`open(...)`) at module level. Copy the `find_chunk_boundaries` function out of
  it instead of importing.
- `make_submission.sh` runs pytest with `|| true` (submission zips even with failures).

## Codebase map

(tiny repo — map inlined; depth cap 2)

| Path | Type | Purpose |
|---|---|---|
| `cs336_basics/` | package | Student implementation target (currently only `__init__.py` with `__version__` and `pretokenization_example.py` with the provided `find_chunk_boundaries` chunker). Implement linear/embedding/RMSNorm/RoPE/attention/SwiGLU/AdamW/tokenizer here. |
| `cs336_basics/__init__.py` | code | Version lookup via `importlib.metadata`. |
| `cs336_basics/pretokenization_example.py` | code | Provided BPE pre-tokenization chunking helper (not importable — see expectations). |
| `tests/` | package | Graded test suite; do not modify except as instructed by the assignment. |
| `tests/adapters.py` | code | The bridge: graded functions call `run_*` adapter functions that the student points at their `cs336_basics` implementation. |
| `tests/test_model.py` | tests | Linear, embedding, RMSNorm, RoPE, SwiGLU, attention, transformer block/LM. |
| `tests/test_tokenizer.py`, `tests/test_train_bpe.py` | tests | BPE training + encode/decode round-trips. |
| `tests/test_optimizer.py`, `tests/test_nn_utils.py` | tests | AdamW, LR schedule, softmax, cross-entropy, gradient clipping. |
| `tests/test_data.py`, `tests/test_serialization.py` | tests | Batching (`get_batch`) and model save/load. |
| `tests/fixtures/`, `tests/_snapshots/` | data | Input corpora (TinyStories sample, GPT-2 vocab/merges) and `.npz` reference outputs. |
| `cs336_spring2025_assignment1_basics.pdf` | docs | Assignment handout — the specification. |
| `make_submission.sh` | script | Gradescope submission packager. |

## Local verification

Prereq: `uv` on PATH (`source ~/.local/bin/env` if freshly installed).

1. `uv sync` → exits 0.
2. `uv run pytest` → 0 collection errors; failures (if repo still unimplemented) are all `NotImplementedError`.
3. `uv run ty check cs336_basics` → "All checks passed!".
4. `uvx ruff check cs336_basics tests` → informational only; upstream starter code has 12 style nits (I001, F402, W605, …) — not a health signal.

## Snapshot

- Snapshot ID: `t9ytei64kkdzbi1d6r9l:default`
- Captured: 2026-10-09T16:55:27.427Z (UTC)
- State captured: uv 0.12.24 installed at `~/.local/bin`, `.venv` synced (Python 3.13.14, torch 2.6.0+cu124), working tree clean.
