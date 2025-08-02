
import torch

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock


class TransformerLM(torch.nn.Module):
    def __init__(self, theta: float, vocab_size: int, num_layers: int, context_length: int, 
            d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = [
            TransformerBlock(theta, context_length, d_model, num_heads, d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None):
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x