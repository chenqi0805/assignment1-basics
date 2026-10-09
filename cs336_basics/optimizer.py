"""AdamW optimizer and cosine learning-rate schedule with warmup.

Both follow the assignment spec: AdamW is Algorithm 1 of Loshchilov and
Hutter (2019) — bias correction folded into an adjusted step size, decoupled
weight decay applied after the gradient step — and the schedule is the LLaMA
cosine annealing with a linear warmup.
"""

from __future__ import annotations

import math

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW with decoupled weight decay, per the spec's Algorithm 1.

    State per parameter: iteration count ``t`` (1-based) and moment estimates
    ``m``, ``v``. The bias correction is folded into the adjusted learning
    rate ``alpha_t = alpha * sqrt(1 - (1-beta2)^t) / (1 - (1-beta1)^t)`` and
    the update divides by ``sqrt(v) + eps`` with the uncorrected ``v``.
    Weight decay ``theta <- theta - alpha * lambda * theta`` runs after the
    gradient step, with the plain (uncorrected) learning rate.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]

                state["t"] += 1
                t = state["t"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                alpha_t = alpha * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                with torch.no_grad():
                    p.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)
                    p.add_(p, alpha=-alpha * weight_decay)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine annealing schedule with linear warmup, as used to train LLaMA.

    - Warmup (``it < warmup_iters``): linear rise, ``(it / warmup_iters) * max``.
    - Annealing (``warmup_iters <= it <= cosine_cycle_iters``): cosine decay
      from ``max`` to ``min``.
    - Post-annealing (``it > cosine_cycle_iters``): constant ``min``.
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
            max_learning_rate - min_learning_rate
        )
    return min_learning_rate
