#!/usr/bin/env python3
"""
Helper script to prepare text data for training by converting it to memory-mapped format.

This script tokenizes text data and saves it as a memory-mapped numpy array for efficient
loading during training.
"""

import argparse
import numpy as np
import os
from cs336_basics.tokenizer import Tokenizer


def prepare_memmap_data(text_file: str, output_file: str, vocab_file: str, merges_file: str):
    """
    Convert text data to tokenized memory-mapped format.
    
    Args:
        text_file: Path to input text file
        output_file: Path to output memory-mapped file
        vocab_file: Path to vocabulary file
        merges_file: Path to merges file
    """
    print(f"Loading tokenizer from {vocab_file} and {merges_file}")
    tokenizer = Tokenizer.from_files(vocab_file, merges_file)
    
    print(f"Reading text from {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("Tokenizing text...")
    tokens = tokenizer.encode(text)
    
    print(f"Tokenized {len(text):,} characters into {len(tokens):,} tokens")
    print(f"Vocabulary size: {len(tokenizer.idx_to_bytes)}")
    
    # Convert to numpy array and save as memory-mapped file
    tokens_array = np.array(tokens, dtype=np.int32)
    
    print(f"Saving to {output_file}")
    # Save as memory-mapped file
    memmap_array = np.memmap(output_file, dtype=np.int32, mode='w+', shape=tokens_array.shape)
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
    
    args = parser.parse_args()
    
    prepare_memmap_data(args.text_file, args.output_file, args.vocab_file, args.merges_file)


if __name__ == "__main__":
    main()
