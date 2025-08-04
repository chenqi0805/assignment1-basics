from jaxtyping import Float, Int
from torch import Tensor, cat, gather, multinomial, sort
import torch

from cs336_basics.softmax import softmax
from cs336_basics.transformer_lm import TransformerLM

def decoding(
        model: TransformerLM,
        temperature: float,
        min_prob: float,
        input_tokens: Int[Tensor, " seq_len"]) -> Int[Tensor, " seq_len"]:
    v = model(input_tokens)
    last_token_v = v[..., -1, :]
    prob = softmax(last_token_v / temperature, dim=-1)
    sorted_prob, sorted_indices = sort(prob, dim=-1, descending=True)
    mask = sorted_prob >= min_prob
    filtered_prob = sorted_prob * mask  # zero out values < min_prob
    filtered_prob = filtered_prob / filtered_prob.sum(dim=-1, keepdim=True)
    # Sample from filtered distribution
    sampled_idx = multinomial(filtered_prob, num_samples=1)  # (..., 1)
    sampled_token = gather(sorted_indices, dim=-1, index=sampled_idx)

    # Shift input tokens left and append predicted token
    output_tokens = cat([input_tokens[..., 1:], sampled_token], dim=-1)  # (..., seq_len)

    return output_tokens

if __name__ == "__main__":
    from cs336_basics.tokenizer import Tokenizer
    from cs336_basics.transformer_lm import TransformerLM

    model = TransformerLM(
        theta=0.5,
        vocab_size=10,
        num_layers=2,
        context_length=4,
        d_model=4,
        num_heads=2,
        d_ff=8,
    )
    input_tokens = torch.tensor([3, 2, 1, 4])
    output_tokens = decoding(model, 1.0, 0.0, input_tokens)
    print(output_tokens)