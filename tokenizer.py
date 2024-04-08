from typing import List

class Tokenizer(object):
    """Tokenise the input by splitting it by space."""

    def __init__(self, vocab_file:str) -> None:
        self.vocab = {}
        self.idx2token = {}
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'

        self.vocab[self.bos_token] = 0
        self.idx2token[0] = ''

        with open(vocab_file, 'r') as f:
            for i, line in enumerate(f):
                self.vocab[line.strip()] = i + 1
                self.idx2token[i + 1] = line.strip()

        self.idx2token[len(self.vocab)] = ''
        self.vocab[self.eos_token] = len(self.vocab)

        self.bos_idx = 0
        self.eos_idx = self.vocab[self.eos_token]

        self.vocab_size = len(self.vocab)

    def encode(self, token:str) -> int:
        return self.vocab[token]

    def decode(self, idx:int) -> str:
        return self.idx2token[idx]
