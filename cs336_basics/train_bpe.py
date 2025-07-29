from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import multiprocessing
import os
from typing import List
import regex as re
import heapq

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _pretokenize(text_parts: list[str]):
    pretokenized = Counter()
    for text_part in text_parts:
        pretokenized.update(re.findall(PAT, text_part))
    return pretokenized

def split_text_and_pretokenize(args: tuple[str, List[str]]) -> Counter:
    text, special_tokens = args
    text_parts = []
    if special_tokens:
        special_pattern = f"(?:{'|'.join(re.escape(s) for s in special_tokens)})"
        text_parts = [part for part in re.split(special_pattern, text) if part and part not in special_tokens]
    else:
        text_parts = [text]
    return _pretokenize(text_parts)


def parallel_tokenize(texts: List[str], special_tokens: List[str], num_processes: int) -> Counter:
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Parallel processing of split + pretokenize for each text
        counters = pool.imap_unordered(split_text_and_pretokenize, [(text, special_tokens) for text in texts])

    # Merge all counters
    final_counter = Counter()
    for c in counters:
        final_counter.update(c)

    return final_counter

class PairItem:
    """自定义类用于在堆中实现正确的排序"""
    def __init__(self, count, bytes1, bytes2):
        self.count = count
        self.bytes1 = bytes1
        self.bytes2 = bytes2
    
    def __lt__(self, other):
        # 首先按频次降序（大的在前）
        if self.count != other.count:
            return self.count > other.count
        # 频次相同时，按第一个token的字节降序
        if self.bytes1 != other.bytes1:
            return self.bytes1 > other.bytes1
        # 第一个token相同时，按第二个token的字节降序
        return self.bytes2 > other.bytes2
    
    def __eq__(self, other):
        return (self.count == other.count and 
                self.bytes1 == other.bytes1 and 
                self.bytes2 == other.bytes2)

class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str], num_processes: int=4):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = num_processes

    def train(self, input_path: str | os.PathLike):
        vocabs = set([bytes([i]) for i in range(256)])
        vocabs.update([s.encode('utf-8') for s in self.special_tokens])
        chunks = []
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)
        print("number of parallel processing chunks:", len(chunks))
        for chunk in chunks:
            print("chunk size:", len(chunk))
        
        pretokenized = parallel_tokenize(chunks, self.special_tokens, self.num_processes)
        token_tuple_count = {tuple(bytes([b]) for b in token.encode('utf-8')): count for token, count in pretokenized.items()}
        token_tuples = list(token_tuple_count.keys())

        print("Pretokenization done. Start invert indexing...")
        inverted_indices = defaultdict(list)
        for i in range(len(token_tuples)):
            token_tuple = token_tuples[i]
            for j in range(len(token_tuple)-1):
                for k in range(j+1, len(token_tuple)):
                    inverted_indices[b''.join(token_tuple[j:k+1])].append(i)
        
        byte_pair_count = Counter()
        for token_tuple, count in token_tuple_count.items():
            for i in range(len(token_tuple)-1):
                byte_pair_count[token_tuple[i:i+2]] += count

        byte_pair_count_max_heap = [PairItem(count, pair[0], pair[1]) for pair, count in byte_pair_count.items()]
        heapq.heapify(byte_pair_count_max_heap)

        print("Start merging...")
        remaining = self.vocab_size - len(vocabs)
        merges = []
        while remaining > 0:
            pairItem = heapq.heappop(byte_pair_count_max_heap)
            tuple_to_merge = (pairItem.bytes1, pairItem.bytes2)
            mergedBytes = b''.join(tuple_to_merge)
            while mergedBytes in vocabs or pairItem.count != byte_pair_count[tuple_to_merge]:
                pairItem = heapq.heappop(byte_pair_count_max_heap)
                tuple_to_merge = (pairItem.bytes1, pairItem.bytes2)
                mergedBytes = b''.join(tuple_to_merge)
            merges.append(tuple_to_merge)
            vocabs.add(mergedBytes)
            remaining -= 1
            for idx in inverted_indices[mergedBytes]:
                old_token_tuple = token_tuples[idx]
                word = b''.join(old_token_tuple).decode('utf-8')
                word_count = pretokenized[word]
                new_token_tuple = self.merge_token_tuple(old_token_tuple, tuple_to_merge)
                token_tuples[idx] = new_token_tuple
                byte_pair_count_before_update = Counter()
                for i in range(len(old_token_tuple)-1):
                    byte_pair_count_before_update[old_token_tuple[i:i+2]] = byte_pair_count[old_token_tuple[i:i+2]]
                for i in range(len(new_token_tuple)-1):
                    byte_pair_count_before_update[new_token_tuple[i:i+2]] = byte_pair_count[new_token_tuple[i:i+2]]
                for i in range(len(old_token_tuple)-1):
                    byte_pair_count[old_token_tuple[i:i+2]] -= word_count
                for i in range(len(new_token_tuple)-1):
                    byte_pair_count[new_token_tuple[i:i+2]] += word_count
                byte_pairs_to_push = set()
                for byte_pair, old_count in byte_pair_count_before_update.items():
                    if old_count != byte_pair_count[byte_pair]:
                        byte_pairs_to_push.add(byte_pair)
                for byte_pair in byte_pairs_to_push:
                    heapq.heappush(byte_pair_count_max_heap, PairItem(byte_pair_count[byte_pair], byte_pair[0], byte_pair[1]))
        vocab_dict = {i: v for i, v in enumerate(vocabs)}
        return vocab_dict, merges

    def merge_token_tuple(self, token_tuple, tuple_to_merge):
        new_token_tuple = []
        idx = 0
        while idx < len(token_tuple):
            if idx < len(token_tuple) - 1 and token_tuple[idx:idx+2] == tuple_to_merge:
                new_token_tuple.append(b''.join(tuple_to_merge))
                idx += 2
            else:
                new_token_tuple.append(token_tuple[idx])
                idx += 1
        return tuple(new_token_tuple)
    
# obj = BPETokenizer(269, ["<|endoftext|>", " "])
# vocab_dict, merges = obj.train("data/test.txt")
# print(merges)
# # print((1, 2, 3)[1:3])
# # print(tuple([s.encode('utf-8') for s in 'ac']))
