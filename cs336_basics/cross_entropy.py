from torch import Tensor
from jaxtyping import Float, Int
import torch

def cross_entropy(
        logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    logsumexp = torch.logsumexp(logits, dim=-1)
    target_logits = torch.gather(logits, 1, targets[:, None]).squeeze(1)
    loss = logsumexp - target_logits
    return loss.mean()