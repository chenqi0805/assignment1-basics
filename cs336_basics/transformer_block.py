import torch

from cs336_basics.multihead_self_attention_with_rope import MultiHeadSelfAttentionWithRoPE
from cs336_basics.positionwise_feedforward import SwiGLU
from cs336_basics.rmsnorm import RMSNorm


class TransformerBlock(torch.nn.Module):
    def __init__(self, theta: float, max_seq_len: int, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn = MultiHeadSelfAttentionWithRoPE(theta, max_seq_len, d_model, num_heads, d_model, device=device)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
