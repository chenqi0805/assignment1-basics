from train_bpe import BPETokenizer
# from train_bpe_no_inverted_indices import BPETokenizer

VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]

obj = BPETokenizer(VOCAB_SIZE, SPECIAL_TOKENS)
vocab_dict, merges = obj.train("data/TinyStoriesV2-GPT4-train.txt")

with open('TinyStoriesV2-merges.txt', 'w', encoding='utf-8') as f:
    for tup in merges:
        f.write(f"{tup}\n")

with open('TinyStoriesV2-vocabs.txt', 'w', encoding='utf-8') as f:
    for key, value in vocab_dict.items():
        f.write(f"{key}: {value}\n")