from typing import List

class Tokenizer(object):
    def __init__(self, vocab_file:str) -> None:
        self.vocab = {}
        self.idx2token = {}

        self.vocab['<bos>'] = 0
        self.idx2token[0] = ''

        with open(vocab_file, 'r') as f:
            for i, line in enumerate(f):
                self.vocab[line.strip()] = i
                self.idx2token[i] = line.strip()

        self.vocab['<eos>'] = len(self.vocab)
        self.idx2token[len(self.vocab)] = ''

        self.vocab_size = len(self.vocab)

    def encode(self, text:str) -> List[int]:
        tokens = text.split()
        return [self.vocab[token] for token in tokens]

    def decode(self, tokens:List[int]) -> str:
        return ' '.join([self.idx2token[token] for token in tokens])
