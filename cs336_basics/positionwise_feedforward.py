from einops import einsum
import torch


class SwiGLU(torch.nn.Module):
    weights1: torch.nn.Parameter
    weights2: torch.nn.Parameter
    weights3: torch.nn.Parameter

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.weights1 = torch.nn.Parameter(torch.empty(self.d_ff, self.d_model, **factory_kwargs))
        torch.nn.init.trunc_normal_(self.weights1, mean=0.0, std=1, a=-3.0, b=3.0)
        self.weights2 = torch.nn.Parameter(torch.empty(self.d_model, self.d_ff, **factory_kwargs))
        torch.nn.init.trunc_normal_(self.weights2, mean=0.0, std=1, a=-3.0, b=3.0)
        self.weights3 = torch.nn.Parameter(torch.empty(self.d_ff, self.d_model, **factory_kwargs))
        torch.nn.init.trunc_normal_(self.weights3, mean=0.0, std=1, a=-3.0, b=3.0)

    def forward(self, x):
        w_1x = einsum(self.weights1, x, "d_ff d_model, ... d_model -> ... d_ff")
        w_3x = einsum(self.weights3, x, "d_ff d_model, ... d_model -> ... d_ff")
        swish = w_1x * torch.sigmoid(w_1x)
        product = einsum(swish, w_3x, "... d_ff, ... d_ff -> ... d_ff")
        result = einsum(self.weights2, product, "d_model d_ff, ... d_ff -> ... d_model")
        return result