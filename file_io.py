import os

from abc import ABC, abstractmethod
from typing import List


class CompressFileIO(ABC):
    def __init__(self, path: str) -> None:
        self.path = path
        self.mode = self._get_mode()
        self.file_handle = self._get_file_handle()

    def close(self) -> None:
        self.file_handle.close()

    @abstractmethod
    def _get_mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _get_file_handle(self):
        raise NotImplementedError


class CompressFileReader(CompressFileIO):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def _get_file_handle(self):
        return open(self.path, self.mode)

    @abstractmethod
    def _get_mode(self) -> str:
        raise NotImplementedError

    def __iter__(self) -> None:
        return self

    @abstractmethod
    def __next__(self) -> int:
        raise NotImplementedError


class BinaryCompressFileReader(CompressFileReader):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.buffer_byte = 0
        self.buffer_n_bits = 0

    def _get_mode(self) -> str:
        return 'rb'

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


class CompressFileWriter(CompressFileIO):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def _get_file_handle(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        return open(self.path, self.mode)

    @abstractmethod
    def _get_mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def write(self, bits: List[int]) -> None:
        raise NotImplementedError


class TextCompressFileWriter(CompressFileWriter):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def _get_mode(self) -> str:
        return 'a'

    def write(self, bits: List[int]) -> None:
        for bit in bits:
            self.file_handle.write(str(bit))


class BinaryCompressFileWriter(CompressFileWriter):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.buffer_byte = 0
        self.buffer_n_bits = 0 # number of bits in the current byte, in [0, 7]

    def _get_mode(self) -> str:
        return 'ab'

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
