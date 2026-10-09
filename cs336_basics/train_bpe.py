"""Byte-level BPE tokenizer training.

Implements the CS336 Assignment 1 training procedure:

1. Split the corpus on special tokens (no merging across those boundaries).
2. Pre-tokenize with the GPT-2 regex pattern to avoid counting pairs across
   pre-token boundaries.
3. Count distinct pre-tokens, then iteratively merge the most frequent pair,
   breaking ties by preferring the lexicographically greater pair.

Pair counts are updated incrementally (only pairs overlapping a merged site
change), using an inverted index from pairs to the pre-tokens that contain
them.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import pairwise

import regex as re

# GPT-2 pre-tokenization pattern (from tiktoken; see the assignment spec).
GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Type alias: a word is a tuple of "tokens", each token being a bytes object
# of length >= 1 (initially single bytes).
Word = tuple[bytes, ...]


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> Iterable[str]:
    """Split text on special tokens so no merges can cross their boundaries."""
    if not special_tokens:
        return [text]
    # re.escape in case a special token contains regex metacharacters like '|'.
    return re.split("|".join(map(re.escape, special_tokens)), text)


def _merge_word(word: Word, pair: tuple[bytes, bytes], merged: bytes) -> Word:
    """Merge all (non-overlapping, left-to-right) occurrences of pair in word."""
    out: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on the text file at input_path.

    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: Maximum final vocabulary size, including the 256 byte
            tokens, tokens produced by merges, and special tokens.
        special_tokens: Special tokens to add to the vocabulary. They are
            never split and never participate in merges.

    Returns:
        vocab: Mapping from token ID to the bytes of that token.
        merges: List of (left, right) byte pairs, ordered by creation.
    """
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    num_merges = max(0, vocab_size - 256 - len(special_tokens))

    # --- Pre-tokenize and count distinct words -------------------------------
    word_counts: Counter[bytes] = Counter()
    for chunk in _split_on_special_tokens(text, special_tokens):
        for match in re.finditer(GPT2_PRETOKENIZE_PATTERN, chunk):
            word_counts[match.group().encode("utf-8")] += 1

    # One entry per distinct word: its token sequence and its corpus frequency.
    word_seqs: list[Word] = []
    word_freqs: list[int] = []
    for word, freq in word_counts.items():
        word_seqs.append(tuple(bytes([b]) for b in word))
        word_freqs.append(freq)

    # --- Initial pair counts and inverted index ------------------------------
    # pair_counts[p] = total corpus occurrences of adjacent token pair p.
    # pair_to_words[p] = indices of words currently containing p.
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for idx, (seq, freq) in enumerate(zip(word_seqs, word_freqs)):
        for pair in pairwise(seq):
            pair_counts[pair] += freq
            pair_to_words[pair].add(idx)

    # --- Merge loop -----------------------------------------------------------
    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        # Highest count wins; ties go to the lexicographically greater pair.
        best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best)
        merged = best[0] + best[1]

        # Update only the words that contain the merged pair.
        for idx in pair_to_words.pop(best, set()):
            old_seq = word_seqs[idx]
            freq = word_freqs[idx]
            for pair in pairwise(old_seq):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            new_seq = _merge_word(old_seq, best, merged)
            for pair in pairwise(new_seq):
                pair_counts[pair] += freq
                pair_to_words[pair].add(idx)
            word_seqs[idx] = new_seq

    # --- Build vocabulary -----------------------------------------------------
    # 256 byte tokens, then merge tokens, then special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, (left, right) in enumerate(merges):
        vocab[256 + i] = left + right
    next_id = 256 + len(merges)
    for special in special_tokens:
        vocab[next_id] = special.encode("utf-8")
        next_id += 1

    return vocab, merges
