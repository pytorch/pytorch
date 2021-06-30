import torch.testing._internal.common_utils as common
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes, onlyCPU
)
from torch.testing import all_types

import torch
import array

TORCH_TO_ARRAYTYPE = {
    torch.int8    : 'b',
    torch.uint8   : 'B',
    torch.int16   : 'h',
    torch.int32   : 'i',
    torch.int64   : 'q',
    torch.float32 : 'f',
    torch.float64 : 'd',
}

def get_dtype_size(dtype):
    if dtype.is_floating_point:
        bytes = torch.finfo(dtype).bits / 8
    else:
        bytes = torch.iinfo(dtype).bits / 8
    return int(bytes)

# Tests for the `frombuffer` function (only work on CPU):
#   Constructs tensors from Python objects that implement the buffer protocol,
#   without copying data.
class TestBufferProtocol(common.TestCase):
    INPUT = [0, 1, 2, 3]

    def _run_test(self, dtype, arr, count=-1, first=0, offset=None, **kwargs):
        if offset is None:
            offset = first * get_dtype_size(dtype)
        last = first + count if count > 0 else len(arr)

        pyarray = array.array(TORCH_TO_ARRAYTYPE[dtype], arr)
        tensor = torch.frombuffer(pyarray, dtype=dtype, count=count, offset=offset, **kwargs)

        self.assertSequenceEqual(pyarray[first:last], tensor)
        return (pyarray, tensor)

    @onlyCPU
    @dtypes(*all_types())
    def test_same_type(self, device, dtype):
        self._run_test(dtype, [1])
        self._run_test(dtype, [0, 1, 2, 3])
        self._run_test(dtype, [0] * 100)

    @onlyCPU
    @dtypes(*all_types())
    def test_requires_grad(self, device, dtype):
        def _run_test_and_check_grad(requires_grad, *args, **kwargs):
            kwargs["requires_grad"] = requires_grad
            _, tensor = self._run_test(*args, **kwargs)
            self.assertTrue(tensor.requires_grad == requires_grad)
        if dtype.is_floating_point or dtype.is_complex:
            _run_test_and_check_grad(True, dtype, [1])
            _run_test_and_check_grad(True, dtype, [0, 1, 2, 3])
            _run_test_and_check_grad(True, dtype, [0] * 100)
        _run_test_and_check_grad(False, dtype, [1])
        _run_test_and_check_grad(False, dtype, [0, 1, 2, 3])
        _run_test_and_check_grad(False, dtype, [0] * 100)

    @onlyCPU
    @dtypes(*all_types())
    def test_with_offset(self, device, dtype):
        input = self.INPUT
        # Offset should be valid whenever there is, at least,
        # one remaining element
        for i in range(len(input)):
            self._run_test(dtype, input, first=i)

    @onlyCPU
    @dtypes(*all_types())
    def test_with_count(self, device, dtype):
        input = self.INPUT
        # Count should be valid for any valid in the interval
        # [-1, len(input)], except for 0
        for i in range(-1, len(input) + 1):
            if i != 0:
                self._run_test(dtype, input, count=i)

    @onlyCPU
    @dtypes(*all_types())
    def test_with_count_and_offset(self, device, dtype):
        input = self.INPUT
        # Explicit default count [-1, 1, 2, ..., len]
        for i in range(-1, len(input) + 1):
            if i != 0:
                self._run_test(dtype, input, count=i)
        # Explicit default offset [0, 1, ..., len - 1]
        for i in range(len(input)):
            self._run_test(dtype, input, first=i)
        # All possible combinations of count and dtype aligned
        # offset for 'input'
        # count:[1, 2, ..., len - 1] x first:[0, 1, ..., len - count]
        for i in range(1, len(input)):
            for j in range(len(input) - i + 1):
                self._run_test(dtype, input, count=i, first=j)

    @onlyCPU
    @dtypes(*all_types())
    def test_invalid_positional_args(self, device, dtype):
        input = self.INPUT
        bytes = get_dtype_size(dtype)
        in_bytes = len(input) * bytes
        # Empty array
        with self.assertRaisesRegex(ValueError,
                                    r"both buffer length \(0\) and count"):
            self._run_test(dtype, [])
        # Count equals 0
        with self.assertRaisesRegex(ValueError,
                                    r"both buffer length .* and count \(0\)"):
            self._run_test(dtype, input, count=0)
        # Offset negative and bigger than total length
        with self.assertRaisesRegex(ValueError,
                                    rf"offset \(-{bytes} bytes\) must be"):
            self._run_test(dtype, input, first=-1)
        with self.assertRaisesRegex(ValueError,
                                    rf"offset \({in_bytes} bytes\) must be .* "
                                    rf"buffer length \({in_bytes} bytes\)"):
            self._run_test(dtype, input, first=len(input))
        # Non-multiple offset with all elements
        if bytes > 1:
            offset = bytes - 1
            with self.assertRaisesRegex(ValueError,
                                        rf"buffer length \({in_bytes - offset} bytes\) after "
                                        rf"offset \({offset} bytes\) must be"):
                self._run_test(dtype, input, offset=bytes - 1)
        # Count too big for each good first element
        for first in range(len(input)):
            count = len(input) - first + 1
            with self.assertRaisesRegex(ValueError,
                                        rf"requested buffer length \({count} \* {bytes} bytes\) "
                                        rf"after offset \({first * bytes} bytes\) must .*"
                                        rf"buffer length \({in_bytes} bytes\)"):
                self._run_test(dtype, input, count=count, first=first)

    @onlyCPU
    @dtypes(*all_types())
    def test_shared_buffer(self, device, dtype):
        input = self.INPUT

        # Modify the whole tensor
        arr, tensor = self._run_test(dtype, input)
        tensor[:] = 5
        self.assertSequenceEqual(arr, tensor)
        self.assertTrue((tensor == 5).all().item())

        # Modify the whole tensor from all valid offsets, given
        # a count value
        for count in range(-1, len(input) + 1):
            if count == 0:
                continue

            actual_count = count if count > 0 else len(input)
            for first in range(len(input) - actual_count):
                last = first + actual_count
                arr, tensor = self._run_test(dtype, input, first=first, count=count)
                tensor[:] = 5
                self.assertSequenceEqual(arr[first:last], tensor)
                self.assertTrue((tensor == 5).all().item())

    @onlyCPU
    @dtypes(*all_types())
    def test_not_a_buffer(self, device, dtype):
        with self.assertRaisesRegex(ValueError,
                                    r"object does not implement Python buffer protocol."):
            torch.frombuffer(self.INPUT, dtype=dtype)

    @onlyCPU
    @dtypes(*all_types())
    def test_non_writable_buffer(self, device, dtype):
        with self.assertWarnsOnceRegex(UserWarning,
                                       r"The given buffer is not writable."):
            torch.frombuffer(b"\x01\x02\x03\x04\x05\x06\x07\x08", dtype=dtype)

instantiate_device_type_tests(TestBufferProtocol, globals())

if __name__ == "__main__":
    common.run_tests()
