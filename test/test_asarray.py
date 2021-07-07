import torch.testing._internal.common_utils as common
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes, onlyCPU,
    skipMeta
)
from torch.testing import all_types, get_all_dtypes
from torch.utils.dlpack import to_dlpack

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

def getaddr(a):
    if isinstance(a, torch.Tensor):
        return a.data_ptr()
    elif isinstance(a, array.array):
        return a.buffer_info()[0]
    raise RuntimeError(f"object {a} not supported.")

def getdevice(a):
    if isinstance(a, torch.Tensor):
        return a.device
    return torch.device("cpu")

def gettensorlike(o, tensor):
    if isinstance(o, torch.Tensor):
        return o
    return torch.tensor(o, dtype=tensor.dtype, device=tensor.device)

class TestAsArray(common.TestCase):
    def _check(self, original, cvt=lambda t: t, is_alias=True, same_dtype=True, same_device=True, **kwargs):
        """Check the output of 'asarray', given its input and assertion informations.

        Besides calling 'asarray' itself, this function does 4 different checks:
            1. Whether the result is aliased or not, depending on 'is_alias'
            2. Whether the result has the expected dtype and elements
            3. Whether the result lives in the expected device
            4. Whether the result has its 'requires_grad' set or not
        """
        result = torch.asarray(cvt(original), **kwargs)
        original_tensor = gettensorlike(original, result)
        self.assertTrue(isinstance(result, torch.Tensor))

        # 1. The storage pointers should be equal only if 'is_alias' is set
        if is_alias:
            self.assertEqual(result.data_ptr(), getaddr(original))
        else:
            self.assertNotEqual(result.data_ptr(), getaddr(original))

        # 2. Comparison of the elements only takes place if the original
        # sequence and the resulting tensor have the same data type
        if same_dtype:
            self.assertEqual(original_tensor, result)
        else:
            dtype = kwargs.get("dtype", torch.get_default_dtype())
            self.assertEqual(original_tensor.shape, result.shape)
            self.assertEqual(dtype, result.dtype)


        # 3. Given the specified target device, we first check whether
        # its type is the same, and then if its index is the same (if it
        # is not None)
        if same_device:
            device = getdevice(original)
        else:
            device = torch.device(kwargs.get("device", "cpu"))

        # Compare the target device type, and its index
        self.assertEqual(device.type, result.device.type)
        if device.index is not None:
            self.assertEqual(device.index, result.device.index)

        # 4. By default, 'requires_grad' is unset
        self.assertEqual(result.requires_grad, kwargs.get("requires_grad", False))

    def _may_compute_grad(self, dtype):
        return dtype.is_floating_point or dtype.is_complex

    # Skipping 'meta' devices, since there's no point in comparing their
    # data pointer (which is basically the point here), since they all
    # return 0.
    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_alias_from_tensor(self, device, dtype):
        # DLpack does not explicitly support bool
        # It does it through uint8 type
        if dtype is torch.bool:
            return

        original = common.make_tensor((5, 5), device, dtype)

        def check(**kwargs):
            self._check(original, **kwargs)

        check(requires_grad=self._may_compute_grad(dtype))
        check(copy=False)
        check(dtype=dtype)
        check(dtype=dtype, copy=False)
        check(device=device)
        check(device=device, dtype=dtype)
        check(device=device, dtype=dtype, copy=False)

    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_copy_tensor(self, device, dtype):
        if dtype is torch.bool:
            return

        original = common.make_tensor((5, 5), device, dtype)

        def check(**kwargs):
            self._check(original, is_alias=False, **kwargs)

        check(requires_grad=self._may_compute_grad(dtype), copy=True)
        check(dtype=dtype, copy=True)
        check(device=device, copy=True)
        check(device=device, dtype=dtype, copy=True)

        # Copy is forced because of different device
        if torch.cuda.is_available():
            other = "cuda" if device == "cpu" else "cpu"
            check(same_device=False, device=other)
            check(same_device=False, device=other, copy=True)
            check(same_device=False, device=other, dtype=dtype, copy=True)

        # Copy is forced because of different dtype
        for other in get_all_dtypes():
            if dtype != other:
                check(same_dtype=False, dtype=other)
                check(same_dtype=False, dtype=other, copy=True)

    # Skipping 'meta', since 'to_dlpack' does not work for them.
    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_alias_from_dlpack(self, device, dtype):
        # DLpack does not explicitly support bool
        # It does it through uint8 type
        if dtype is torch.bool:
            return

        original = common.make_tensor((5, 5), device, dtype)

        def check(**kwargs):
            self._check(original, to_dlpack, **kwargs)

        check(requires_grad=self._may_compute_grad(dtype))
        check(copy=False)
        check(dtype=dtype)
        check(dtype=dtype, copy=False)
        check(device=device)
        check(device=device, dtype=dtype)
        check(device=device, dtype=dtype, copy=False)

    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_copy_from_dlpack(self, device, dtype):
        if dtype is torch.bool:
            return

        original = common.make_tensor((5, 5), device, dtype)

        def check(**kwargs):
            self._check(original, to_dlpack, is_alias=False, **kwargs)

        check(requires_grad=self._may_compute_grad(dtype), copy=True)
        check(dtype=dtype, copy=True)
        check(device=device, copy=True)
        check(device=device, dtype=dtype, copy=True)

        # Copy is forced because of different device
        if torch.cuda.is_available():
            other = "cuda" if device == "cpu" else "cpu"
            check(same_device=False, device=other)
            check(same_device=False, device=other, copy=True)
            check(same_device=False, device=other, dtype=dtype, copy=True)

        # Copy is forced because of different dtype
        for other in get_all_dtypes():
            if dtype != other:
                check(same_dtype=False, dtype=other)
                check(same_dtype=False, dtype=other, copy=True)

    @onlyCPU
    @dtypes(*all_types())
    def test_alias_from_buffer(self, device, dtype):
        original = common.make_tensor((5,), device, dtype)
        arr = array.array(TORCH_TO_ARRAYTYPE[dtype], original)

        def check(**kwargs):
            self._check(arr, **kwargs)

        check(dtype=dtype, requires_grad=self._may_compute_grad(dtype))
        check(dtype=dtype, copy=False)
        check(device=device, dtype=dtype)
        check(device=device, dtype=dtype, copy=False)

    @onlyCPU
    @dtypes(*all_types())
    def test_copy_from_buffer(self, device, dtype):
        original = common.make_tensor((5,), device, dtype)
        arr = array.array(TORCH_TO_ARRAYTYPE[dtype], original)

        def check(**kwargs):
            self._check(arr, is_alias=False, **kwargs)

        check(dtype=dtype, requires_grad=self._may_compute_grad(dtype), copy=True)
        check(dtype=dtype, device=device, copy=True)

        # Copy is forced because of different device
        if torch.cuda.is_available():
            check(same_device=False, dtype=dtype, device="cuda")
            check(same_device=False, dtype=dtype, device="cuda", copy=True)

    @dtypes(*get_all_dtypes())
    def test_copy_list(self, device, dtype):
        original = common.make_tensor((5, 5), device, dtype)

        def check(**kwargs):
            self._check(original, torch.Tensor.tolist, is_alias=False, **kwargs)

        check(device=device, dtype=dtype)
        check(device=device, dtype=dtype, requires_grad=self._may_compute_grad(dtype))
        check(device=device, dtype=dtype, copy=True)

    @dtypes(*get_all_dtypes())
    def test_unsupported_alias(self, device, dtype):
        original = common.make_tensor((5, 5), device, dtype)

        if torch.cuda.is_available():
            other_device = "cuda" if device == "cpu" else "cpu"

            with self.assertRaisesRegex(ValueError,
                                        f"from device '{device}' to '{other_device}'"):
                torch.asarray(original, device=other_device, copy=False)

        for other_dtype in get_all_dtypes():
            if other_dtype != dtype:
                with self.assertRaisesRegex(ValueError,
                                            "with dtype '.*' into dtype '.*'"):
                    torch.asarray(original, dtype=other_dtype, copy=False)

        with self.assertRaisesRegex(ValueError,
                                    "can't alias arbitrary sequence"):
            torch.asarray(original.tolist(), copy=False)

instantiate_device_type_tests(TestBufferProtocol, globals())
instantiate_device_type_tests(TestAsArray, globals())

if __name__ == "__main__":
    common.run_tests()
