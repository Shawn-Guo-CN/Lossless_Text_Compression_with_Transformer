import os

from abc import ABC, abstractmethod
from typing import List


class CompressFileIO(ABC):
    def __init__(self, path: str, mode: str) -> None:
        self.path = path
        self.mode = mode
        self.file_handle = self._get_file_handle()

    def __iter__(self):
        return self

    def _get_file_handle(self):
        if ('a' in self.mode or 'w' in self.mode) and \
        os.path.exists(self.path):
            os.remove(self.path)
        return open(self.path, self.mode)

    def close(self) -> None:
        self.file_handle.close()


class TextCompressFileIO(CompressFileIO):
    def __init__(self, path: str, mode: str) -> None:
        super().__init__(path, mode)

    def write(self, bits: List[int]) -> None:
        for bit in bits:
            self.file_handle.write(str(bit))

    def __next__(self) -> int:
        tmp_byte = self.file_handle.read(1)
        if len(tmp_byte) == 0:
            raise StopIteration
        return int(tmp_byte)


class BinaryCompressFileIO(CompressFileIO):
    def __init__(self, path: str, mode: str) -> None:
        super().__init__(path, mode)
        self.buffer_byte = 0
        self.buffer_n_bits = 0

    def __next__(self) -> int:
        if self.buffer_n_bits == 0:
            tmp_byte = self.file_handle.read(1)
            if len(tmp_byte) == 0:
                raise StopIteration
            self.buffer_byte = tmp_byte[0]
            self.buffer_n_bits = 8
        assert self.buffer_n_bits > 0
        self.buffer_n_bits -= 1
        return (self.buffer_byte >> self.buffer_n_bits) & 1

    def write(self, bits: List[int]) -> None:
        for bit in bits:
            assert bit == 0 or bit == 1, "Invalid bit value."
            self.write_bit(bit)

    def write_bit(self, bit:int) -> None:
        self.buffer_byte = (self.buffer_byte << 1) | bit
        self.buffer_n_bits += 1
        if self.buffer_n_bits == 8:
            self.file_handle.write(bytes([self.buffer_byte]))
            self.buffer_byte = 0
            self.buffer_n_bits = 0

    def close(self) -> None:
        while self.buffer_n_bits != 0:
            self.write_bit(0)
        super().close()
