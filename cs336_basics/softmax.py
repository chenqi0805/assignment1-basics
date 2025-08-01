import torch

def softmax(x: torch.Tensor, dim: int):
    # for numerical stability, subtract max along dim
    x_max = torch.amax(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)