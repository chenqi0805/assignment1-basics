from einops import einsum, rearrange
import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        factory_kwargs = {"device": device}
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires even dimension size"
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute inverse frequency: [d_k/2]
        positions = torch.arange(max_seq_len, **factory_kwargs).float()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, **factory_kwargs).float() / d_k))
        sinusoid = einsum(positions, inv_freq, "max_seq_len, half_d_k -> max_seq_len half_d_k")

        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)

        # Register non-persistent buffers
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)

        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        rot_mat = rearrange(
            torch.stack([cos, -sin, sin, cos], dim=0),
            '(r c) ... -> ... r c',
            r=2, c=2
        )

        # Split even/odd channels
        x_pair = rearrange(x, "... seq_len (d_pair two) -> ... seq_len d_pair two", two = 2)

        # rotate "i j, j -> i"
        x_rot = einsum(rot_mat, x_pair, "... d_pair i j, ... d_pair j -> ... d_pair i")
        out = rearrange(x_rot, "... seq_len d_pair two -> ... seq_len (d_pair two)", two = 2)
        
        return  out.to(in_type)