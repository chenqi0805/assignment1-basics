import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        torch.nn.init.trunc_normal_(self.embedding, mean=0.0, std=1, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]