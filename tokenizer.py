from typing import Dict


class Tokenizer(object):
    """Tokenise the input by splitting it by space."""
    bos_token = '<bos>'
    eos_token = '<eos>'
    unk_token = '<unk>'

    def __init__(self, vocab_file:str) -> None:
        self.vocab = {}
        self.idx2token = {}

        for i, (k, v) in enumerate(
            self.get_special_token_inverse_mappings().items()
        ):
            self.vocab[k] = i
            self.idx2token[i] = v

        self.vocab[self.unk_token] = len(self.vocab)
        self.idx2token[len(self.vocab) - 1] = self.unk_token

        with open(vocab_file, 'r') as f:
            for line in f.readlines():
                if len(line.strip()) == 0: continue
                self.vocab[line.strip()] = len(self.vocab)
                self.idx2token[len(self.vocab) - 1] = line.strip()

        self.eos_idx = self.vocab[self.eos_token]
        self.vocab_size = len(self.vocab)

    def encode(self, token:str) -> int:
        try:
            return self.vocab[token]
        except KeyError:
            print(f'Unknown token: {token}')
            return self.vocab[self.unk_token]

    def decode(self, idx:int) -> str:
        return self.idx2token[idx]

    @staticmethod
    def get_special_token_mappings() -> Dict[str, str]:
        return {
            '\\n', '<new_line>',
            ' ', ' <space> '
        }

    @classmethod
    def get_special_token_inverse_mappings(cls) -> Dict[str, str]:
        return {
            '<new_line>': '\n',
            '<space>': ' ',
            cls.bos_token: '',
            cls.eos_token: '',
        }
