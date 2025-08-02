import math
from typing import Callable, Optional
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            betas = group["betas"]
            weight_decay = group["weight_decay"] # Get weight decay.
            eps = group["eps"] # Get epsilon.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p.data)) # Get first moment vector from the state, or initial value.
                v = state.get("v", torch.zeros_like(p.data)) # Get second moment vector from the state, or initial value."
                alpha_t = state.get("alpha_t", lr) # Get alpha_t from the state, or initial value.")
                grad = p.grad.data # Get the gradient of loss with respect to p.
                m = betas[0] * m + (1.0 - betas[0]) * grad.data # Update biased first moment estimate.
                v = betas[1] * v + (1.0 - betas[1]) * grad.data ** 2 # Update biased second raw moment estimate.
                alpha_t = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t) # Compute alpha_t.
                p.data -= alpha_t * m / (torch.sqrt(v) + eps) # Update parameter.
                p.data -= lr * weight_decay * p.data # Update weight tensor in-place.
                state["m"] = m # Update state."
                state["v"] = v
                state["alpha_t"] = alpha_t
                state["t"] = t + 1 # Increment iteration number.
        return loss