import ast
import itertools
import json
import os
import pathlib
import resource
import psutil
import regex as re
from typing import Iterable, Iterator, Optional

from tests.common import gpt2_bytes_to_unicode

GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def pretokenize(text: str) -> list[bytes]:
    str_tokens = re.findall(GPT2_SPLIT_PATTERN, text)
    byte_tokens = [s.encode("utf-8") for s in str_tokens]
    return byte_tokens

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        self.special_tokens = special_tokens or []
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        self.special_tokens_set = set(self.special_tokens)
        self.special_tokens_bytes = [
            token.encode("utf-8") for token in self.special_tokens_set
        ]
        self.idx_to_bytes = {idx: token_bytes for idx, token_bytes in vocab.items()}
        offset = max(self.idx_to_bytes.keys()) + 1
        vocabs = set(self.idx_to_bytes.values())
        special_tokens_to_be_added = [token for token in self.special_tokens_bytes if token not in vocabs]
        self.idx_to_bytes.update({i + offset: token_bytes for i, token_bytes in enumerate(special_tokens_to_be_added)})
        self.bytes_to_idx = {b: idx for idx, b in self.idx_to_bytes.items()}
        self.merges = merges

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        vocab = _read_bytes_dict(vocab_filepath)
        merges = _read_byte_pairs(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens_sorted)})"
            text_parts = [part for part in re.split(special_pattern, text) if part]
        else:
            text_parts = [text]
        pretokenized = self._pretokenize(text_parts)
        pretokenized_set = set(pretokenized)
        token_tuple_map = {word: tuple(bytes([b]) for b in word.encode('utf-8')) for word in pretokenized_set if word not in self.special_tokens_set}

        for pair in self.merges:
            merged_pair = b''.join(pair)
            for word, token_tuple in token_tuple_map.items():
                new_token_tuple = []
                i = 0
                while i < len(token_tuple):
                    if i < len(token_tuple) - 1 and token_tuple[i:i+2] == pair:
                        new_token_tuple.append(merged_pair)
                        i += 2
                    else:
                        new_token_tuple.append(token_tuple[i])
                        i += 1
                token_tuple_map[word] = tuple(new_token_tuple)

        token_idx_map = {word: [self.bytes_to_idx[token] for token in token_tuple] for word, token_tuple in token_tuple_map.items()}
        token_idx_map.update({token: [self.bytes_to_idx[token.encode('utf-8')]] for token in self.special_tokens_set})
        token_indices = [token_idx_map[word] for word in pretokenized]

        return list(itertools.chain.from_iterable(token_indices))
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        byte_list = [self.idx_to_bytes[idx] for idx in ids]
        text = b''.join(byte_list).decode('utf-8', errors='replace')
        return text
    
    def _pretokenize(self, text_parts: list[str]):
        pretokenized = []
        for text_part in text_parts:
            if text_part in self.special_tokens_set:
                pretokenized.append(text_part)
                continue
            str_tokens = re.findall(GPT2_SPLIT_PATTERN, text_part)
            pretokenized.extend(str_tokens)
        return pretokenized
    
def _read_bytes_dict(filepath: str) -> dict[int, bytes]:
    result = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key_str, val_str = line.split(":", 1)
            key = int(key_str.strip())
            value = ast.literal_eval(val_str.strip())
            result[key] = value
    return result

def _read_byte_pairs(filepath: str) -> list[tuple[bytes, bytes]]:
    byte_pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = ast.literal_eval(line)
            byte_pairs.append(pair)
    return byte_pairs

# obj = Tokenizer(
#     vocab={0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'},
#     merges=[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')],
#     special_tokens=[]
# )

# def memory_limit(max_mem):
#     def decorator(f):
#         def wrapper(*args, **kwargs):
#             process = psutil.Process(os.getpid())
#             prev_limits = resource.getrlimit(resource.RLIMIT_AS)
#             resource.setrlimit(
#                 resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1)
#             )
#             try:
#                 result = f(*args, **kwargs)
#                 return result
#             finally:
#                 # Even if the function above fails (e.g., it exceeds the
#                 # memory limit), reset the memory limit back to the
#                 # previous limit so other tests aren't affected.
#                 resource.setrlimit(resource.RLIMIT_AS, prev_limits)

#         return wrapper

#     return decorator

# @memory_limit(int(1e6))
# def _encode_iterable(tokenizer, iterable):
#     """
#     We place tokenizer.encode_iterable into a separate function so we can limit memory
#     for just this function. We set the memory limit to 1MB.
#     """
#     yield from tokenizer.encode_iterable(iterable)

# FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "../tests/fixtures"
# VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
# MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

# def get_tokenizer_from_vocab_merges_path(
#     vocab_path: str | os.PathLike,
#     merges_path: str | os.PathLike,
#     special_tokens: Optional[list[str]] = None,
# ):
#     gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
#     with open(vocab_path) as vocab_f:
#         gpt2_vocab = json.load(vocab_f)
#     gpt2_bpe_merges = []
#     with open(merges_path) as f:
#         for line in f:
#             cleaned_line = line.rstrip()
#             if cleaned_line and len(cleaned_line.split(" ")) == 2:
#                 gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
#     # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
#     # just return the original bytes, so we don't force students to use
#     # any particular encoding scheme.
#     vocab = {
#         gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
#         for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
#     }
#     # If any of the special tokens don't exist in the vocab, append them to the vocab.
#     if special_tokens:
#         for special_token in special_tokens:
#             byte_encoded_special_token = special_token.encode("utf-8")
#             if byte_encoded_special_token not in set(vocab.values()):
#                 vocab[len(vocab)] = byte_encoded_special_token

#     merges = [
#         (
#             bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
#             bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
#         )
#         for merge_token_1, merge_token_2 in gpt2_bpe_merges
#     ]
#     return get_tokenizer(vocab, merges, special_tokens)

# def get_tokenizer(
#     vocab: dict[int, bytes],
#     merges: list[tuple[bytes, bytes]],
#     special_tokens: Optional[list[str]] = None,
# ):
#     """Given a vocabulary, a list of merges, and a list of special tokens,
#     return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

#     Args:
#         vocab: dict[int, bytes]
#             The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
#             to bytes (token bytes)
#         merges: list[tuple[bytes, bytes]]
#             BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
#             representing that <token1> was merged with <token2>.
#             Merges are ordered by order of creation.
#         special_tokens: Optional[list[str]]
#             A list of string special tokens for the tokenizer. These strings will never
#             be split into multiple tokens, and will always be kept as a single token.

#     Returns:
#         A BPE tokenizer that uses the provided vocab, merges, and special tokens.
#     """
#     return Tokenizer(
#         vocab=vocab,
#         merges=merges,
#         special_tokens=special_tokens
#     )

# tokenizer = get_tokenizer_from_vocab_merges_path(
#         vocab_path=VOCAB_PATH,
#         merges_path=MERGES_PATH,
#     )

# with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
#     ids = []
#     for _id in _encode_iterable(tokenizer, f):
#         ids.append(_id)