import math
from einops import einsum, rearrange
import torch

from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_in: int):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.w_q = torch.nn.Parameter(torch.empty(num_heads, self.d_k, d_in))
        torch.nn.init.trunc_normal_(self.w_q, mean=0.0, std=1, a=-3.0, b=3.0)
        self.w_k = torch.nn.Parameter(torch.empty(num_heads, self.d_k, d_in))
        torch.nn.init.trunc_normal_(self.w_k, mean=0.0, std=1, a=-3.0, b=3.0)
        self.w_v = torch.nn.Parameter(torch.empty(num_heads, self.d_v, d_in))
        torch.nn.init.trunc_normal_(self.w_v, mean=0.0, std=1, a=-3.0, b=3.0)
        self.w_o = torch.nn.Parameter(torch.empty(d_model, self.d_v * num_heads))
        torch.nn.init.trunc_normal_(self.w_o, mean=0.0, std=1, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor): # x: (..., seq_len, d_in)
        assert self.d_in == x.shape[-1]
        query = einsum(self.w_q, x, "num_heads d_k d_in, ... seq_len d_in -> ... num_heads seq_len d_k")
        key = einsum(self.w_k, x, "num_heads d_k d_in, ... seq_len d_in -> ... num_heads seq_len d_k")
        value = einsum(self.w_v, x, "num_heads d_v d_in, ... seq_len d_in -> ... num_heads seq_len d_v")
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        multihead = scaled_dot_product_attention(query, key, value, mask) # (... num_heads seq_len d_v)
        multihead = rearrange(multihead, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        multihead_self_attention = einsum(multihead, self.w_o, "... seq_len d, d_out d -> ... seq_len d_out")
        return multihead_self_attention