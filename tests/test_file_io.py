import os

from file_io import BinaryCompressFileReader, BinaryCompressFileWriter


def test_binary_file_io1():
    tmp_file = './tmp_binary_file_io1.bin'

    test_input_1 = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

    writer = BinaryCompressFileWriter(tmp_file)
    writer.write(test_input_1)
    writer.close()

    reader = BinaryCompressFileReader(tmp_file)
    test_output_1 = [bit for bit in reader]
    reader.close()

    assert test_input_1 == test_output_1

    os.remove(tmp_file)


def test_binary_file_io2():
    tmp_file = './tmp_binary_file_io2.bin'

    test_input_2 = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

    writer = BinaryCompressFileWriter(tmp_file)
    writer.write(test_input_2)
    writer.close()

    reader = BinaryCompressFileReader(tmp_file)
    test_output_2 = [bit for bit in reader]
    reader.close()

    assert test_input_2  + [0] * 4 == test_output_2

    os.remove(tmp_file)
