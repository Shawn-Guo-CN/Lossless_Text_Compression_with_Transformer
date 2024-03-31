from tokenizer import Tokenizer

class DataLoader(object):
    def __init__(
        self,
        data_path: str,
        tokenizer:Tokenizer,
        max_len: int, # should <= the block size of the model
    ) -> None:
        # NOTE: we assume the data file is just one line
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(data_path, 'r') as f:
            self.data = f.readline().strip().split()
        self.target = self.data + [tokenizer.eos_token]
        self.data = [tokenizer.bos_token] + self.data

        self.idx = 0
        self.x = []
        self.y = []

    def __iter__(self):
        return self
 
    def __next__(self):
        if self.idx >= len(self.data):
            raise StopIteration
        else:
            self.x.append(self.tokenizer.encode(self.data[self.idx]))
            self.y.append(self.tokenizer.encode(self.target[self.idx]))

            self.idx += 1

            self.x = self.x[-self.max_len:]
            self.y = self.y[-self.max_len:]
            return {
                'x': self.x,
                'y': self.y
            }
