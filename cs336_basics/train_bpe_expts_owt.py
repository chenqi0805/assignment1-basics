from train_bpe import BPETokenizer
# from train_bpe_no_inverted_indices import BPETokenizer

VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<|endoftext|>"]

obj = BPETokenizer(VOCAB_SIZE, SPECIAL_TOKENS)
vocab_dict, merges = obj.train("data/owt_train.txt")

with open('owt-merges.txt', 'w', encoding='utf-8') as f:
    for tup in merges:
        f.write(f"{tup}\n")

with open('owt-vocabs.txt', 'w', encoding='utf-8') as f:
    for key, value in vocab_dict.items():
        f.write(f"{key}: {value}\n")