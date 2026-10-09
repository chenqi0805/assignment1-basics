"""Transformer LM building blocks: Linear, Embedding, RMSNorm, SwiGLU, RoPE,
causal multi-head self-attention, the pre-norm Transformer block, and the full
Transformer language model (spec §3.1–§3.6)."""

import math

import torch
from einops import rearrange
from torch import nn

from .nn_utils import silu, softmax

__all__ = [
    "Embedding",
    "Linear",
    "MultiheadSelfAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "TransformerBlock",
    "TransformerLM",
    "scaled_dot_product_attention",
]


class Linear(nn.Module):
    """Bias-free linear transformation y = x @ W.T.

    The weight is stored as ``W`` with shape (d_out, d_in) for memory-ordering
    reasons (spec §3.4.2); the transpose happens in the multiply.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory = {"device": device, "dtype": dtype}
        std = math.sqrt(2.0 / (in_features + out_features))
        weight = torch.empty(out_features, in_features, **factory)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.W = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


class Embedding(nn.Module):
    """Token embedding lookup (spec §3.4.3), implemented without nn.Embedding."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory = {"device": device, "dtype": dtype}
        weight = torch.empty(num_embeddings, embedding_dim, **factory)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (spec §3.5.1, eq. 4).

    Computes in float32 to avoid overflow when squaring, then downcasts back.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight.to(torch.float32)
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    """Position-wise feed-forward network with SwiGLU activation (spec §3.5.2, eq. 7):

    FFN(x) = W2(SiLU(W1 x) ⊙ W3 x), no biases.

    d_ff defaults to approximately 8/3 · d_model, rounded down to a multiple of
    64 so the inner layer uses hardware well.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = (8 * d_model // 3) // 64 * 64
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embeddings (spec §3.5.3, Su et al. 2021).

    Rotates adjacent feature pairs (x_0, x_1), (x_2, x_3), ... of the query/key
    vectors; pair k (0-based) at position i rotates by angle i·Θ^(−2k/d_k).
    cos/sin values are precomputed for up to max_seq_len positions in
    non-persistent buffers (no learnable parameters).

    Inputs may have an arbitrary number of leading batch dimensions.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        pair_index = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        # (max_seq_len, d_k // 2): angle for each position and adjacent pair.
        angles = positions[:, None] * theta ** (-2.0 * pair_index / d_k)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        # (..., seq_len, d_k // 2), broadcast against any leading batch dims.
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
        ).flatten(-2)
        return rotated.to(x.dtype)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention (spec §3.5.4, eq. 11).

    Q, K: (batch..., seq, d_k); V: (batch..., seq, d_v). ``mask`` is boolean
    with True meaning the query attends to the key; False positions receive a
    -inf pre-softmax score so their attention probability is exactly zero.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, dim=-1) @ V


class MultiheadSelfAttention(nn.Module):
    """Causal multi-head self-attention (spec §3.5.5, eq. 14).

    d_k = d_v = d_model / num_heads. Q, K, V come from three separate Linear
    projections; heads are split so d_k acts as a batch dimension for
    attention; an optional RoPE rotates Q and K (never V) using
    ``token_positions``; a causal (lower-triangular) mask prevents attending
    to future positions.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        rope_max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.use_rope = use_rope
        if use_rope:
            if rope_max_seq_len is None:
                raise ValueError("rope_max_seq_len is required when use_rope=True")
            self.rope = RotaryPositionalEmbedding(
                rope_theta, self.d_k, rope_max_seq_len, device=device
            )

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        q = rearrange(self.q_proj(x), "b s (h dk) -> b h s dk", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b s (h dk) -> b h s dk", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b s (h dk) -> b h s dk", h=self.num_heads)
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        causal_mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=x.device
        ).tril()
        out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        out = rearrange(out, "b h s dk -> b s (h dk)")
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (spec §3.5/§3.6, eq. 15):

    y = x + MHA(RMSNorm(x)); z = y + FFN(RMSNorm(y)).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        rope_max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model,
            num_heads,
            use_rope=use_rope,
            rope_theta=rope_theta,
            rope_max_seq_len=rope_max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model (spec §3.1, Figure 1):

    token embedding → num_layers pre-norm blocks (with RoPE) → final RMSNorm →
    bias-free LM head producing next-token logits.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int | None = None,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff=d_ff,
                    use_rope=True,
                    rope_theta=rope_theta,
                    rope_max_seq_len=context_length,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for block in self.blocks:
            x = block(x, token_positions)
        return self.lm_head(self.ln_final(x))
