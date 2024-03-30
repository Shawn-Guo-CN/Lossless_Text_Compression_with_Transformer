from tokenizer import Tokenizer

class DataLoader(object):
    def __init__(
        self,
        data_path: str,
        tokenizer:Tokenizer,
        max_len: int, # should <= the block size of the model
    ) -> None:
        # NOTE: we assume the data file is just one line
        with open(data_path, 'r') as f:
            self.data = f.readline().strip().split()
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.idx = 0
        self.batch = []

    def __iter__(self):
        return self
 
    def __next__(self):
        if self.idx == 0:
            self.batch.append(self.tokenizer.encode('<bos>'))
            self.idx += 1
            return self.batch
        elif self.idx == len(self.data) + 1:
            raise StopIteration
        else:
            self.batch.append(self.tokenizer.encode(self.data[self.idx - 1]))
            self.idx += 1
            self.batch = self.batch[-self.max_len:]
            return self.batch
