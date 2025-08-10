#!/usr/bin/env python3
"""
Helper script to prepare text data for training by converting it to memory-mapped format.

This script tokenizes text data and saves it as a memory-mapped numpy array for efficient
loading during training.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from typing import Iterable, List
import numpy as np
import os

from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer

def find_batch(texts: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch = []
    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def encode_batch(tokenizer: Tokenizer, batch: List[str]) -> List[int]:
    return list(tokenizer.encode_iterable(batch))

def parallel_encode(tokenizer: Tokenizer, texts: Iterable[str], batch_size: int, num_processes: int) -> List[str]:
    # results = []
    # with ProcessPoolExecutor(max_workers=num_processes) as exe:
    #     futures = []
    #     for batch in find_batch(texts, batch_size):
    #         futures.append(exe.submit(encode_batch, tokenizer, batch))
    #     for fut in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing"):
    #         arr = fut.result()
    #         results.extend(arr)
    # return results
            
    encode_with_tokenizer = partial(encode_batch, tokenizer)
    with multiprocessing.Pool(processes=num_processes) as pool:
        token_id_batches = pool.imap(encode_with_tokenizer, find_batch(texts, batch_size))

        # Merge all
        final_token_ids = []
        for batch in token_id_batches:
            final_token_ids.extend(batch)

        return final_token_ids

def prepare_memmap_data(text_file: str, output_file: str, vocab_file: str, merges_file: str, batch_size: int, num_processes: int):
    """
    Convert text data to tokenized memory-mapped format.
    
    Args:
        text_file: Path to input text file
        output_file: Path to output memory-mapped file
        vocab_file: Path to vocabulary file
        merges_file: Path to merges file
    """
    print(f"Loading tokenizer from {vocab_file} and {merges_file}")
    tokenizer = Tokenizer.from_files(vocab_file, merges_file, ["<|endoftext|>"])
    
    print(f"Reading text from {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        print("Tokenizing text...")
        tokens = parallel_encode(tokenizer, f, batch_size, num_processes)
    
    # print(f"Tokenized {len(text):,} characters into {len(tokens):,} tokens")
    print(f"Vocabulary size: {len(tokenizer.idx_to_bytes)}")
    
    # Convert to numpy array and save as memory-mapped file
    tokens_array = np.array(tokens, dtype=np.uint16)
    
    print(f"Saving to {output_file}")
    # Save as memory-mapped file
    memmap_array = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=tokens_array.shape)
    memmap_array[:] = tokens_array
    memmap_array.flush()
    
    print(f"Successfully saved {len(tokens):,} tokens to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Prepare text data for training")
    parser.add_argument("--text_file", type=str, required=True,
                       help="Path to input text file")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to output memory-mapped file")
    parser.add_argument("--vocab_file", type=str, required=True,
                       help="Path to vocabulary file")
    parser.add_argument("--merges_file", type=str, required=True,
                       help="Path to merges file")
    parser.add_argument("--batch_size", type=int, default=10000,
                       help="Batch size for tokenization")
    parser.add_argument("--num_processes", type=int, default=8,
                       help="Number of processes to use for tokenization")
    
    args = parser.parse_args()
    
    prepare_memmap_data(
        args.text_file, 
        args.output_file, 
        args.vocab_file, 
        args.merges_file,
        args.batch_size,
        args.num_processes)


if __name__ == "__main__":
    main()
