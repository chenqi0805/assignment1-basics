"""Byte-level BPE tokenizer.

A tokenizer over the vocabulary and merges produced by `train_bpe`.
Encoding splits on special tokens first (so they stay whole and nothing
merges across them), pre-tokenizes each remaining piece with the GPT-2
regex, then applies BPE merges in order of creation (equivalently: always
merge the present pair with the lowest creation rank).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from itertools import pairwise

import regex as re

from cs336_basics.train_bpe import GPT2_PRETOKENIZE_PATTERN, _merge_word

# Type alias for a pair of byte tokens.
Pair = tuple[bytes, bytes]


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte-to-unicode-string map.

    Maps every byte to a printable unicode character (printable bytes map to
    themselves; others, including space, map to codepoints 256+), so byte
    sequences can round-trip through text files like JSON and space-separated
    merges lists.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    """Byte-level BPE tokenizer with optional special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ):
        """
        Args:
            vocab: Mapping from token ID to the bytes of that token.
            merges: BPE merges ordered by creation; earlier pairs have priority.
            special_tokens: Special tokens kept whole during encoding. Tokens
                are matched longest-first, so overlapping specials resolve to
                the longest match.
        """
        self.vocab: dict[int, bytes] = dict(vocab)
        self.vocab_inv: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merges_ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        # Append special tokens that aren't in the vocab yet, as the spec
        # requires, so they get IDs and round-trip through decode.
        for special in self.special_tokens:
            special_bytes = special.encode("utf-8")
            if special_bytes not in self.vocab_inv:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = special_bytes
                self.vocab_inv[special_bytes] = new_id
        # Longest-first alternating split pattern; group 1 captures the special
        # token so re.split keeps it in the output.
        self._special_split_pattern = (
            "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            if self.special_tokens
            else None
        )
        self.special_token_ids: dict[str, int] = {
            special: self.vocab_inv[special.encode("utf-8")]
            for special in self.special_tokens
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load a Tokenizer from files in the GPT-2 serialization format.

        vocab_filepath: JSON object mapping each token's remapped string
            (via bytes_to_unicode, so non-UTF-8 bytes survive) to its ID.
        merges_filepath: one merge per line, "left right" in remapped
            strings; blank/malformed lines are skipped.
        """
        byte_decoder = {c: b for b, c in bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab: dict[str, int] = json.load(f)
        vocab = {
            int(token_id): bytes(byte_decoder[ch] for ch in token_str)
            for token_str, token_id in raw_vocab.items()
        }
        merges: list[Pair] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip()
                if not cleaned:
                    continue
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    continue
                merges.append(
                    (
                        bytes(byte_decoder[ch] for ch in parts[0]),
                        bytes(byte_decoder[ch] for ch in parts[1]),
                    )
                )
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs. Special tokens stay whole.

        Not memory-bounded: the input and output are held in full.
        """
        ids: list[int] = []
        for piece, is_special in self._split_on_special(text):
            if is_special:
                token_id = self.special_token_ids.get(piece)
                if token_id is None:
                    raise ValueError(f"Special token not in vocab: {piece!r}")
                ids.append(token_id)
            else:
                for match in re.finditer(GPT2_PRETOKENIZE_PATTERN, piece):
                    ids.extend(self._encode_word(match.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g. an open file), yielding
        one token ID at a time so large inputs stream in constant memory."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token IDs back to text, replacing malformed UTF-8 with U+FFFD."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def _split_on_special(self, text: str) -> Iterator[tuple[str, bool]]:
        """Yield (piece, is_special) chunks; specials never cross piece
        boundaries, so nothing merges across them."""
        if self._special_split_pattern is None:
            yield text, False
            return
        for i, piece in enumerate(re.split(self._special_split_pattern, text)):
            if piece:
                yield piece, i % 2 == 1

    def _encode_word(self, word: bytes) -> list[int]:
        """BPE-encode one pre-token: repeatedly merge the present pair with
        the lowest creation rank (ties among pairs can't occur; ranks are
        unique), scanning occurrences left-to-right."""
        tokens: list[bytes] = [bytes([b]) for b in word]
        while len(tokens) > 1:
            best_pair: Pair | None = None
            best_rank: int | None = None
            for pair in pairwise(tokens):
                rank = self.merges_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            tokens = list(
                _merge_word(tuple(tokens), best_pair, best_pair[0] + best_pair[1])
            )
        return [self.vocab_inv[token] for token in tokens]
