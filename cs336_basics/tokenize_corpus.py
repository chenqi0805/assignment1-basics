"""Tokenize a corpus with a BPE tokenizer and write memmap-able .npy arrays.

One-off preprocessing for ``cs336_basics.train`` (spec §5.3): train a BPE
tokenizer (or reuse serialized vocab/merges files), serialize it in the GPT-2
format ``Tokenizer.from_files`` reads, encode the corpus, and save the token
stream as two ``.npy`` files (train and a small validation tail) loadable with
``np.memmap``.

Example:
    python -m cs336_basics.tokenize_corpus \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 --output-dir data/tinystories
"""

import argparse
import array
import json
import os

import numpy as np

from .tokenizer import Tokenizer, bytes_to_unicode
from .train_bpe import train_bpe


def _save_tokenizer_gpt2(tokenizer: Tokenizer, vocab_path: str, merges_path: str) -> None:
    """Serialize a Tokenizer's raw-byte vocab and merges in GPT-2 format."""
    byte_map = bytes_to_unicode()
    token_map = {}
    for token_id, token_bytes in tokenizer.vocab.items():
        token_map["".join(byte_map[b] for b in token_bytes)] = token_id
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(token_map, f)
    with open(merges_path, "w", encoding="utf-8") as f:
        f.writelines(
            "".join(byte_map[b] for b in left)
            + " "
            + "".join(byte_map[b] for b in right)
            + "\n"
            for left, right in tokenizer.merges_ranks
        )


def _encode_stream(tokenizer: Tokenizer, input_path: str) -> array.array:
    """Encode a corpus memory-safely, streaming the file line by line."""
    ids = array.array("I")
    with open(input_path, encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Corpus text file to tokenize")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens; leave empty for none",
    )
    parser.add_argument(
        "--vocab-file",
        help="Reuse an existing GPT-2-format vocab instead of training BPE",
    )
    parser.add_argument(
        "--merges-file",
        help="Reuse an existing GPT-2-format merges file instead of training BPE",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--val-fraction", type=float, default=0.001)
    args = parser.parse_args(argv)

    special_tokens = args.special_tokens or []
    os.makedirs(args.output_dir, exist_ok=True)
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    merges_path = os.path.join(args.output_dir, "merges.txt")

    if args.vocab_file and args.merges_file:
        tokenizer = Tokenizer.from_files(
            args.vocab_file, args.merges_file, special_tokens=special_tokens
        )
        print(f"Loaded tokenizer from {args.vocab_file}")
    else:
        print(f"Training BPE tokenizer (vocab size {args.vocab_size}) on {args.input}")
        vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens)
        tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)
        _save_tokenizer_gpt2(tokenizer, vocab_path, merges_path)
        print(f"Saved tokenizer to {vocab_path} and {merges_path}")

    print(f"Encoding {args.input}")
    ids = _encode_stream(tokenizer, args.input)
    num_val = max(2, int(len(ids) * args.val_fraction))
    val_ids = ids[-num_val:]
    train_ids = ids[:-num_val]

    dtype = np.uint16 if args.vocab_size <= 65535 else np.uint32
    train_path = os.path.join(args.output_dir, "train_tokens.npy")
    val_path = os.path.join(args.output_dir, "val_tokens.npy")
    np.save(train_path, np.frombuffer(train_ids, dtype=np.uint32).astype(dtype))
    np.save(val_path, np.frombuffer(val_ids, dtype=np.uint32).astype(dtype))
    print(
        f"Wrote {len(train_ids):,} train tokens to {train_path} and "
        f"{len(val_ids):,} validation tokens to {val_path} (dtype {dtype.__name__})"
    )


if __name__ == "__main__":
    main()
