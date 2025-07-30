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
    
class ByteInWordNode:
    def __init__(self, bytes: bytes, word_freq: int):
        self.bytes = bytes
        self.word_freq = word_freq
        self.prev = None
        self.next = None

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

        print("Pretokenization done. Start invert indexing...")

        pair_to_nodes = defaultdict(set)
        byte_pair_count = Counter()
        for word, word_count in pretokenized.items():
            token_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
            prevNode = None
            for j in range(len(token_tuple)-1):
                pair = token_tuple[j:j+2]
                byte_pair_count[pair] += word_count
                currNode = ByteInWordNode(token_tuple[j], word_count)
                pair_to_nodes[pair].add(currNode)
                if prevNode is not None:
                    prevNode.next = currNode
                currNode.prev = prevNode
                prevNode = currNode
            currNode = ByteInWordNode(token_tuple[-1], word_count)
            if prevNode is not None:
                prevNode.next = currNode
            currNode.prev = prevNode

        byte_pair_count_max_heap = [PairItem(count, pair[0], pair[1]) for pair, count in byte_pair_count.items()]
        heapq.heapify(byte_pair_count_max_heap)

        print("Start merging...")
        remaining = self.vocab_size - len(vocabs)
        merges = []
        while remaining > 0:
            if len(byte_pair_count_max_heap) == 0:
                break
            pairItem = heapq.heappop(byte_pair_count_max_heap)
            tuple_to_merge = (pairItem.bytes1, pairItem.bytes2)
            mergedBytes = b''.join(tuple_to_merge)
            if mergedBytes in vocabs or pairItem.count != byte_pair_count[tuple_to_merge]:
                continue
            merges.append(tuple_to_merge)
            vocabs.add(mergedBytes)
            remaining -= 1
            candidate_byte_pairs_to_push = set()
            tuple_to_merge_impacted_nodes = set(pair_to_nodes[tuple_to_merge])
            for node in tuple_to_merge_impacted_nodes:
                if node not in pair_to_nodes[tuple_to_merge]:
                    continue
                prevNode = node.prev
                nextNode = node.next
                nextNextNode = nextNode.next
                if prevNode is not None:
                    byte_pair_count[(prevNode.bytes, node.bytes)] -= prevNode.word_freq
                    byte_pair_count[(prevNode.bytes, mergedBytes)] += prevNode.word_freq
                    pair_to_nodes[(prevNode.bytes, node.bytes)].discard(prevNode)
                    pair_to_nodes[(prevNode.bytes, mergedBytes)].add(prevNode)
                    candidate_byte_pairs_to_push.add((prevNode.bytes, node.bytes))
                    candidate_byte_pairs_to_push.add((prevNode.bytes, mergedBytes))
                if nextNextNode is not None:
                    byte_pair_count[(nextNode.bytes, nextNextNode.bytes)] -= nextNode.word_freq
                    byte_pair_count[(mergedBytes, nextNextNode.bytes)] += node.word_freq
                    pair_to_nodes[(nextNode.bytes, nextNextNode.bytes)].discard(nextNode)
                    pair_to_nodes[(mergedBytes, nextNextNode.bytes)].add(node)
                    candidate_byte_pairs_to_push.add((nextNode.bytes, nextNextNode.bytes))
                    candidate_byte_pairs_to_push.add((mergedBytes, nextNextNode.bytes))
                    nextNextNode.prev = node
                node.bytes = mergedBytes
                node.next = nextNextNode
            del pair_to_nodes[tuple_to_merge]
            del byte_pair_count[tuple_to_merge]

            for byte_pair in candidate_byte_pairs_to_push:
                heapq.heappush(byte_pair_count_max_heap, PairItem(byte_pair_count[byte_pair], byte_pair[0], byte_pair[1]))
                
        vocab_dict = {i: v for i, v in enumerate(vocabs)}
        return vocab_dict, merges
    
# obj = BPETokenizer(269, ["<|endoftext|>", " "])
# vocab_dict, merges = obj.train("data/test.txt")
# print(merges)
# # print((1, 2, 3)[1:3])
# # print(tuple([s.encode('utf-8') for s in 'ac']))
