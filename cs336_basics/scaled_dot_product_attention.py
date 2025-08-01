from einops import einsum
import torch

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    qk = einsum(query, key, "... queries d_k, ... keys d_k -> ... queries keys") / key.shape[-1] ** 0.5
    if mask is not None:
        qk = qk.masked_fill(~mask, float('-inf'))
    qk = softmax(qk, dim=-1)
    return einsum(qk, value, "... queries keys, ... keys d_v -> ... queries d_v")