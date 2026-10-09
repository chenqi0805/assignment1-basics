"""Checkpoint serialization for model and optimizer state (spec §5.2)."""

from typing import IO, Any

import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | IO[bytes] | IO[str],
) -> None:
    """Save model state, optimizer state, and iteration number to ``out``.

    ``out`` may be a path (str/os.PathLike) or a binary file-like object.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | IO[bytes] | IO[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model and optimizer state from ``src``; return the saved iteration."""
    checkpoint: dict[str, Any] = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
