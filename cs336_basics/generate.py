"""Text generation with temperature scaling and top-p sampling (spec §6,
problem decoding).

``decode`` repeatedly samples the next token from the model's last-position
distribution (optionally temperature-scaled and nucleus-truncated) and
appends it, stopping at an end-of-sequence token or a user-specified token
budget. Run as a module to generate from a checkpoint:

    python -m cs336_basics.generate --checkpoint ckpt.pt \
        --vocab-file data/tinystories/vocab.json \
        --merges-file data/tinystories/merges.txt \
        --prompt "Once upon a time" --max-new-tokens 256 --top-p 0.9
"""

import argparse

import torch

from .checkpoint import load_checkpoint
from .model import TransformerLM
from .nn_utils import softmax
from .tokenizer import Tokenizer


def decode(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    context_length: int | None = None,
    device: str = "cpu",
) -> str:
    """Sample a completion for ``prompt`` from the model (spec §6).

    Args:
        model: The TransformerLM to sample from (used in eval mode).
        tokenizer: Tokenizes the prompt and decodes the sampled stream.
        prompt: The prompt text to condition on.
        max_new_tokens: Stop after generating this many tokens even if no
            end-of-sequence token appears.
        temperature: Softmax temperature (eq. 24); ``tau -> 0`` approaches
            greedy (argmax) decoding.
        top_p: Optional nucleus (top-p) threshold: sample only from the
            smallest set of most-probable tokens with cumulative probability
            at least ``p``.
        context_length: Crop the running sequence to this many most-recent
            tokens before each step (None uses the model's RoPE context).
        device: Device to run the model on.

    Returns:
        The prompt plus the sampled completion, decoded.
    """
    model.eval()
    eos_id = tokenizer.special_token_ids.get("<|endoftext|>")
    if context_length is None:
        context_length = max(
            block.attn.rope.max_seq_len if hasattr(block.attn, "rope") else 10**6
            for block in model.blocks
        )
    tokens = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(
                torch.tensor(tokens[-context_length:], device=device)[None, :]
            )[0, -1]
            if temperature <= 0.0:
                next_id = int(torch.argmax(logits))
            else:
                probs = softmax(logits / temperature, dim=-1)
                if top_p is not None:
                    probs = _nucleus_mask(probs, top_p)
                next_id = int(torch.multinomial(probs, num_samples=1))
            tokens.append(next_id)
            if next_id == eos_id:
                break
    return tokenizer.decode(tokens)


def _nucleus_mask(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out all but the nucleus: the smallest set of most-probable tokens
    whose cumulative probability reaches ``top_p``."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    num_keep = int((cumulative < top_p).sum()) + 1  # first index crossing p is kept
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_indices[:num_keep], sorted_probs[:num_keep])
    return filtered / filtered.sum()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab-file", required=True)
    parser.add_argument("--merges-file", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--special-tokens", nargs="*", default=["<|endoftext|>"], help="Empty for none"
    )
    parser.add_argument("--device", default="cpu")
    # Model hyperparameters must match the checkpoint being loaded.
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    args = parser.parse_args(argv)

    tokenizer = Tokenizer.from_files(
        args.vocab_file, args.merges_file, special_tokens=args.special_tokens or []
    )
    device = torch.device(args.device)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    # Generation needs no optimizer state; a fresh dummy optimizer carries the
    # load. `optimizer.state_dict()` on a fresh optimizer is empty.
    load_checkpoint(args.checkpoint, model, torch.optim.SGD(model.parameters(), lr=0.0))
    text = decode(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    print(text)


if __name__ == "__main__":
    main()
