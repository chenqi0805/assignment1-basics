from torch import Tensor
from jaxtyping import Float, Int
import torch

def cross_entropy(
        logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    logits_max = torch.amax(logits, dim=-1, keepdim=True)
    logits_centered = logits - logits_max
    logits_exp = torch.exp(logits_centered)
    logits_exp_sum = logits_exp.sum(dim=-1)
    log_logits_exp_sum = torch.log(logits_exp_sum)
    summation = - logits_centered[torch.arange(logits.shape[0]), targets] + log_logits_exp_sum
    return summation.mean()