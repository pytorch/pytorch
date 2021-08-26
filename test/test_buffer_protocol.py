import torch.testing._internal.common_utils as common
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes
)

import torch
import numpy

def get_dtype_size(dtype):
    return int(torch.empty((), dtype=dtype).element_size())

SIZE = 5
SHAPE = (SIZE,)

# Tests for the `frombuffer` function (only work on CPU):
#   Constructs tensors from Python objects that implement the buffer protocol,
#   without copying data.
class TestBufferProtocol(common.TestCase):
    def _run_test(self, shape, dtype, count=-1, first=0, offset=None, **kwargs):
        numpy_dtype = common.torch_to_numpy_dtype_dict[dtype]

        if offset is None:
            offset = first * get_dtype_size(dtype)

        numpy_original = make_tensor(shape, torch.device("cpu"), dtype).numpy()
        original = memoryview(numpy_original)
        # First call PyTorch's version in case of errors.
        # If this call exits successfully, the NumPy version must also do so.
        torch_frombuffer = torch.frombuffer(original, dtype=dtype, count=count, offset=offset, **kwargs)
        numpy_frombuffer = numpy.frombuffer(original, dtype=numpy_dtype, count=count, offset=offset)

        self.assertEqual(numpy_frombuffer, torch_frombuffer)
        self.assertEqual(numpy_frombuffer.__array_interface__["data"][0], torch_frombuffer.data_ptr())
        return (numpy_original, torch_frombuffer)

    @dtypes(*common.torch_to_numpy_dtype_dict.keys())
    def test_same_type(self, device, dtype):
        self._run_test((), dtype)
        self._run_test((4,), dtype)
        self._run_test((10, 10), dtype)

    @dtypes(*common.torch_to_numpy_dtype_dict.keys())
    def test_requires_grad(self, device, dtype):
        def _run_test_and_check_grad(requires_grad, *args, **kwargs):
            kwargs["requires_grad"] = requires_grad
            _, tensor = self._run_test(*args, **kwargs)
            self.assertTrue(tensor.requires_grad == requires_grad)

        requires_grad = dtype.is_floating_point or dtype.is_complex
        _run_test_and_check_grad(requires_grad, (), dtype)
        _run_test_and_check_grad(requires_grad, (4,), dtype)
        _run_test_and_check_grad(requires_grad, (10, 10), dtype)
        _run_test_and_check_grad(False, (), dtype)
        _run_test_and_check_grad(False, (4,), dtype)
        _run_test_and_check_grad(False, (10, 10), dtype)

    @dtypes(*common.torch_to_numpy_dtype_dict.keys())
    def test_with_offset(self, device, dtype):
        # Offset should be valid whenever there is, at least,
        # one remaining element
        for i in range(SIZE):
            self._run_test(SHAPE, dtype, first=i)

    @dtypes(*common.torch_to_numpy_dtype_dict.keys())
    def test_with_count(self, device, dtype):
        # Count should be valid for any valid in the interval
        # [-1, len(input)], except for 0
        for i in range(-1, SIZE + 1):
            if i != 0:
                self._run_test(SHAPE, dtype, count=i)

    @dtypes(*common.torch_to_numpy_dtype_dict.keys())
    def test_with_count_and_offset(self, device, dtype):
        # Explicit default count [-1, 1, 2, ..., len]
        for i in range(-1, SIZE + 1):
            if i != 0:
                self._run_test(SHAPE, dtype, count=i)
        # Explicit default offset [0, 1, ..., len - 1]
        for i in range(SIZE):
            self._run_test(SHAPE, dtype, first=i)
        # All possible combinations of count and dtype aligned
        # offset for 'input'
        # count:[1, 2, ..., len - 1] x first:[0, 1, ..., len - count]
        for i in range(1, SIZE):
            for j in range(SIZE - i + 1):
                self._run_test(SHAPE, dtype, count=i, first=j)

    @dtypes(*common.torch_to_numpy_dtype_dict.keys())
    def test_invalid_positional_args(self, device, dtype):
        bytes = get_dtype_size(dtype)
        in_bytes = SIZE * bytes
        # Empty array
        with self.assertRaisesRegex(ValueError,
                                    r"both buffer length \(0\) and count"):
            empty = numpy.array([])
            torch.frombuffer(empty, dtype=dtype)
        # Count equals 0
        with self.assertRaisesRegex(ValueError,
                                    r"both buffer length .* and count \(0\)"):
            self._run_test(SHAPE, dtype, count=0)
        # Offset negative and bigger than total length
        with self.assertRaisesRegex(ValueError,
                                    rf"offset \(-{bytes} bytes\) must be"):
            self._run_test(SHAPE, dtype, first=-1)
        with self.assertRaisesRegex(ValueError,
                                    rf"offset \({in_bytes} bytes\) must be .* "
                                    rf"buffer length \({in_bytes} bytes\)"):
            self._run_test(SHAPE, dtype, first=SIZE)
        # Non-multiple offset with all elements
        if bytes > 1:
            offset = bytes - 1
            with self.assertRaisesRegex(ValueError,
                                        rf"buffer length \({in_bytes - offset} bytes\) after "
                                        rf"offset \({offset} bytes\) must be"):
                self._run_test(SHAPE, dtype, offset=bytes - 1)
        # Count too big for each good first element
        for first in range(SIZE):
            count = SIZE - first + 1
            with self.assertRaisesRegex(ValueError,
                                        rf"requested buffer length \({count} \* {bytes} bytes\) "
                                        rf"after offset \({first * bytes} bytes\) must .*"
                                        rf"buffer length \({in_bytes} bytes\)"):
                self._run_test(SHAPE, dtype, count=count, first=first)

    @dtypes(*common.torch_to_numpy_dtype_dict.keys())
    def test_shared_buffer(self, device, dtype):
        x = make_tensor((1,), device, dtype)
        # Modify the whole tensor
        arr, tensor = self._run_test(SHAPE, dtype)
        tensor[:] = x
        self.assertEqual(arr, tensor)
        self.assertTrue((tensor == x).all().item())

        # Modify the whole tensor from all valid offsets, given
        # a count value
        for count in range(-1, SIZE + 1):
            if count == 0:
                continue

            actual_count = count if count > 0 else SIZE
            for first in range(SIZE - actual_count):
                last = first + actual_count
                arr, tensor = self._run_test(SHAPE, dtype, first=first, count=count)
                tensor[:] = x
                self.assertEqual(arr[first:last], tensor)
                self.assertTrue((tensor == x).all().item())

                # Modify the first value in the array
                arr[first] = x.item() - 1
                self.assertEqual(arr[first:last], tensor)

    @dtypes(*common.torch_to_numpy_dtype_dict.keys())
    def test_not_a_buffer(self, device, dtype):
        with self.assertRaisesRegex(ValueError,
                                    r"object does not implement Python buffer protocol."):
            torch.frombuffer([1, 2, 3, 4], dtype=dtype)

    @dtypes(*common.torch_to_numpy_dtype_dict.keys())
    def test_non_writable_buffer(self, device, dtype):
        numpy_arr = make_tensor((1,), device, dtype).numpy()
        byte_arr = numpy_arr.tobytes()
        with self.assertWarnsOnceRegex(UserWarning,
                                       r"The given buffer is not writable."):
            torch.frombuffer(byte_arr, dtype=dtype)

    def test_byte_to_int(self):
        byte_array = numpy.array([-1, 0, 0, 0, -1, 0, 0, 0], dtype=numpy.byte)
        tensor = torch.frombuffer(byte_array, dtype=torch.int32)
        self.assertEqual(tensor.numel(), 2)
        # Assuming little endian machine
        self.assertSequenceEqual(tensor, [255, 255])

instantiate_device_type_tests(TestBufferProtocol, globals(), only_for="cpu")

if __name__ == "__main__":
    common.run_tests()
