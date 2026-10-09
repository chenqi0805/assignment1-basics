"""Byte-level BPE tokenizer.

A tokenizer over the vocabulary and merges produced by `train_bpe`.
Encoding splits on special tokens first (so they stay whole and nothing
merges across them), pre-tokenizes each remaining piece with the GPT-2
regex, then applies BPE merges in order of creation (equivalently: always
merge the present pair with the lowest creation rank).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import pairwise

import regex as re

from cs336_basics.train_bpe import GPT2_PRETOKENIZE_PATTERN, _merge_word

# Type alias for a pair of byte tokens.
Pair = tuple[bytes, bytes]


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
        self.vocab = dict(vocab)
        self.merges_ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        # Longest-first alternating split pattern; group 1 captures the special
        # token so re.split keeps it in the output.
        self._special_split_pattern = (
            "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            if self.special_tokens
            else None
        )
        # bytes -> id inversion (used to emit IDs for encoded pieces).
        self.vocab_inv: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.special_token_ids: dict[str, int] = {
            special: self.vocab_inv[special.encode("utf-8")]
            for special in self.special_tokens
            if special.encode("utf-8") in self.vocab_inv
        }

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
