"""Numerical utilities for transformer training.

Stability notes: softmax and cross-entropy both shift inputs by their maximum
before exponentiating (the softmax is invariant to a constant shift, so this
is exact, not an approximation), and cross-entropy cancels log/exp so no full
row of exponentials is ever formed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, Int


def softmax(in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    """Apply softmax along ``dim``.

    Numerically stable: shifts inputs so the max along ``dim`` is 0 before
    exponentiating, so large-magnitude inputs cannot overflow.
    """
    shifted = in_features - in_features.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def silu(x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
    """SiLU (Swish) activation: x · σ(x). Smooth at zero, unlike ReLU."""
    return x * torch.sigmoid(x)


def cross_entropy(
    inputs: Float[torch.Tensor, " batch vocab_size"],
    targets: Int[torch.Tensor, " batch"],
) -> Float[torch.Tensor, ""]:
    """Cross-entropy loss between logits and target indices, averaged over rows.

    Batch-like dimensions come first; the last dimension of ``inputs`` is the
    vocabulary. Equivalent to ``-log softmax(inputs)[target]`` per row without
    exponentiating the row: with the max subtracted, the loss is
    ``logsumexp(shifted) - shifted[target]``.
    """
    flat_logits = inputs.reshape(-1, inputs.size(-1))
    flat_targets = targets.reshape(-1)
    shifted = flat_logits - flat_logits.max(dim=-1, keepdim=True).values
    log_z = shifted.logsumexp(dim=-1)
    target_logits = shifted.gather(-1, flat_targets.unsqueeze(-1)).squeeze(-1)
    return (log_z - target_logits).mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip gradients in place so their global L2 norm is at most ``max_l2_norm``.

    Matches torch.nn.utils.clip_grad_norm_ semantics: params with no gradient
    are skipped, the norm is computed across all gradients jointly, and if it
    exceeds the threshold every gradient is scaled by
    ``max_l2_norm / (total_norm + 1e-6)``.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.norm(2) ** 2 for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(scale)


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample language-modeling windows from a 1D array of token IDs.

    Start indices are uniform over the valid range so every window of
    ``context_length + 1`` tokens fits. Returns (inputs, labels) of shape
    ``(batch_size, context_length)`` where labels are inputs shifted by one.
    """
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offsets = np.arange(context_length)
    x = dataset[starts[:, None] + offsets]
    y = dataset[starts[:, None] + 1 + offsets]
    return (
        torch.as_tensor(x, dtype=torch.int64, device=device),
        torch.as_tensor(y, dtype=torch.int64, device=device),
    )
