from einops import einsum
import torch

class RMSNorm(torch.nn.Module):
    weights: torch.nn.Parameter
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weights = torch.nn.Parameter(torch.empty(d_model, **factory_kwargs))
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=1, a=-3.0, b=3.0)

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        vec = x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = einsum(vec, self.weights, "... d_model, d_model -> ... d_model")
        return result.to(in_dtype)