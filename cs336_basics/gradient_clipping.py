from typing import Iterable

import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    total_norm = torch.norm(torch.stack([
        p.grad.detach().norm() for p in parameters if p.grad is not None
    ]))

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(scale)