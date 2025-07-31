from einops import einsum
import torch


class Linear(torch.nn.Module):
    w: torch.nn.Parameter

    def __init__(self, in_features, out_features, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        torch.nn.init.trunc_normal_(self.w, mean=0.0, std=1, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.w, "... in_features, out_features in_features -> ... out_features")
        return y