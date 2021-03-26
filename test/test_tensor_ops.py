import functools
import torch

from torch.testing._internal.common_utils import (
    TestCase, run_tests, make_tensor, suppress_warnings)

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyOnCPUAndCUDA, ops)

from torch.testing._internal.common_methods_invocations import (
    OpInfo, SampleInput)


class TensorCreationOpInfo(OpInfo):
    """OpInfo Base class for tensor creation operations.
    """

    all_dtypes = torch.testing._dispatch_dtypes(torch.testing.get_all_dtypes())
    all_dtypes_numpycompat = torch.testing._dispatch_dtypes(torch.testing.get_all_dtypes(
        include_half=False,
        include_bfloat16=False,
        include_complex32=False))
    all_dtypes_grad = torch.testing._dispatch_dtypes(torch.testing.floating_and_complex_types())

    def __init__(self,
                 name,
                 *,
                 dtypes=None,
                 supports_autograd=None,
                 **kwargs):
        supports_autograd = False if supports_autograd is None else supports_autograd
        if supports_autograd:
            dtypes = self.all_dtypes_grad if dtypes is None else dtypes
        else:
            dtypes = self.all_dtypes if dtypes is None else dtypes
        super().__init__(name,
                         dtypes=dtypes,
                         supports_autograd=supports_autograd,
                         check_batched_grad=False,
                         check_batched_gradgrad=False,
                         **kwargs)

    def get_devices(self, skip_device=None, max_device_indices=2):
        """Generator of existing device instances except skip_device.
        """
        if isinstance(skip_device, str):
            skip_device = torch.device(skip_device)
        for device_name in torch.testing.get_all_device_types():
            if device_name == 'cuda':
                for index in range(min(max_device_indices, torch.cuda.device_count())):
                    device = torch.device(device_name, index)
                    if skip_device != device:
                        yield device
            else:
                device = torch.device(device_name)
                if skip_device != device:
                    yield device

    def get_convertible_dtypes(self, dtype):
        """Generate dtypes that allow lossless conversion from dtype and then
        back to dtype.
        """
        for types in [
                [torch.complex32, torch.complex64, torch.complex128],
                [torch.float64, torch.complex128],
                [torch.float32, torch.float64, torch.complex64, torch.complex128],
                [torch.float16, torch.float32, torch.float64, torch.complex32, torch.complex64, torch.complex128],
                [torch.bfloat16, torch.float16, torch.float32, torch.float64, torch.complex32, torch.complex64, torch.complex128],
                [torch.int8, torch.int16, torch.int32, torch.int64],
                [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64],
                [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]]:
            if dtype in types:
                for dt in reversed(types):
                    yield dt
                    if dt == dtype:
                        return
        assert 0, dtype


def tensor_variants(device, dtype, requires_grad):
    """Generator of random tensors with variable size and strides.
    """
    for discontiguous in [False, True]:
        if discontiguous and dtype in [torch.bfloat16, torch.float16]:
            # Skipping to avoid the following failures:
            #   RuntimeError: "index_select" not implemented for 'BFloat16'
            #   RuntimeError: "index_select" not implemented for 'Half'
            continue
        for size in [(5,), (3, 4)]:
            yield make_tensor(size, device, dtype,
                              requires_grad=requires_grad,
                              discontiguous=discontiguous)


def sample_inputs_tensor(op_info, device, dtype, requires_grad, pin_memory=False):
    """Generator of samples for torch.tensor and torch.as_tensor.
    """
    if isinstance(device, str):
        device = torch.device(device)

    assert op_info.op in [torch.tensor, torch.as_tensor]

    if op_info.op == torch.tensor:
        kwargs = dict(dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)
    elif op_info.op == torch.as_tensor:
        assert not requires_grad
        kwargs = dict(dtype=dtype, device=device)
    else:
        raise NotImplementedError(op_info.op)

    # op always copies input data
    always_copies = op_info.op in [torch.tensor]

    for r_index, r in enumerate(tensor_variants(device, dtype, requires_grad)):
        # create from output Tensor
        yield SampleInput(r, ref=r, kwargs={}, extra=dict(may_share_memory=not always_copies))
        if pin_memory:
            if r_index == 0:
                yield SampleInput(r, ref=RuntimeError("Can't pin tensor constructed from a variable"), kwargs=kwargs)
        else:
            yield SampleInput(r, ref=r, kwargs=kwargs, extra=dict(may_share_memory=not always_copies))

        # create from identical Tensor
        # checking for shared memory will ensure that clone is not leaking
        yield SampleInput(r.clone(), ref=r, kwargs={}, extra=dict(may_share_memory=False))
        if pin_memory:
            if r_index == 0:
                yield SampleInput(r.clone(), ref=RuntimeError("Can't pin tensor constructed from a variable"), kwargs=kwargs)
        else:
            yield SampleInput(r.clone(), ref=r, kwargs=kwargs, extra=dict(may_share_memory=False))

        # create from identical Tensor using another device
        # checking for shared memory will ensure that to(another_device) is not leaking
        for other_device in op_info.get_devices(skip_device=device):
            obj = r.to(device=other_device)
            if pin_memory:
                if r_index == 0:
                    yield SampleInput(obj, ref=RuntimeError("Can't pin tensor constructed from a variable"), kwargs=kwargs)
            else:
                yield SampleInput(obj, ref=r, kwargs=kwargs, extra=dict(may_share_memory=False))

        # create from list, not checking for shared memory as it is
        # not possible in between a Tensor and a Python list
        yield SampleInput(r.tolist(), ref=r, kwargs=kwargs, extra=dict(enable_pin_test=(pin_memory and device.type == 'cpu')))

        # create from an object that implements sequence protocol
        class Sequence:
            def __init__(self, tensor):
                self.tensor = tensor

            def __len__(self):
                return len(self.tensor)

            def __getitem__(self, key):
                if self.tensor.ndim > 1:
                    return type(self)(self.tensor[key])
                return self.tensor[key]

        yield SampleInput(Sequence(r), ref=r, kwargs=kwargs)

        if dtype in op_info.all_dtypes_numpycompat:

            class ArrayInterfaceAttribute:
                def __init__(self, tensor):
                    self.__array_interface__ = tensor.__array__().__array_interface__

            class ArrayInterfaceProperty:
                def __init__(self, tensor):
                    self.tensor = tensor

                @property
                def __array_interface__(self):
                    return self.tensor.__array__().__array_interface__

            class ArrayInterfaceMethod:
                def __init__(self, tensor):
                    self.tensor = tensor

                def __array__(self):
                    return self.tensor.__array__()

            class CudaArrayInterfaceAttribute:
                def __init__(self, tensor):
                    self.__cuda_array_interface__ = tensor.__cuda_array_interface__

            class CudaArrayInterfaceProperty:
                def __init__(self, tensor):
                    self.tensor = tensor

                @property
                def __cuda_array_interface__(self):
                    return self.tensor.__cuda_array_interface__

            convertible_types = list(op_info.get_convertible_dtypes(dtype))

            for obj_dtype_index, obj_dtype in enumerate(op_info.all_dtypes_numpycompat):

                # skip equality test when conversion dtype->obj_dtype->dtype looses information
                skip_equal_test = obj_dtype not in convertible_types

                may_share_memory = r.device.type == 'cpu' and obj_dtype == dtype
                extra = dict(may_share_memory=may_share_memory and not always_copies,
                             skip_equal_test=skip_equal_test)

                # create from NumPy ndarray
                obj = r.detach().to(dtype=obj_dtype, device='cpu').numpy()
                if pin_memory:
                    if r_index == obj_dtype_index == 0:
                        yield SampleInput(obj, ref=RuntimeError("Can't pin tensor constructed from numpy"), kwargs=kwargs)
                else:
                    yield SampleInput(obj, ref=r, kwargs=kwargs, extra=extra)

                # create from memoryview, PEP 3118 Buffer Protocol
                if 0:
                    # TODO: requires memoryview support for torch.tensor
                    # and fixing point 4 in
                    # https://github.com/pytorch/pytorch/issues/51156#issuecomment-791990731
                    yield SampleInput(memoryview(obj), ref=r, kwargs=kwargs, extra=extra)

                # create from object implementing CPU Array Interface
                if pin_memory:
                    if r_index == obj_dtype_index == 0:
                        yield SampleInput(ArrayInterfaceAttribute(obj),
                                          ref=RuntimeError("Can't pin tensor constructed from __array_interface__"),
                                          kwargs=kwargs, extra=extra)
                        yield SampleInput(ArrayInterfaceMethod(obj),
                                          ref=RuntimeError("Can't pin tensor constructed from __array__"),
                                          kwargs=kwargs, extra=extra)
                else:
                    yield SampleInput(ArrayInterfaceAttribute(obj), ref=r, kwargs=kwargs, extra=extra)
                    yield SampleInput(ArrayInterfaceProperty(obj), ref=r, kwargs=kwargs, extra=extra)
                    yield SampleInput(ArrayInterfaceMethod(obj), ref=r, kwargs=kwargs, extra=extra)

                # create from object implementing CUDA Array Interface
                if r.device.type == 'cuda' and torch.bool not in [obj_dtype, dtype]:
                    obj = r.detach().to(dtype=obj_dtype)
                    may_share_memory = obj_dtype == dtype
                    extra = dict(may_share_memory=may_share_memory and not always_copies, skip_equal_test=skip_equal_test)
                    if pin_memory:
                        if r_index == obj_dtype_index == 0:
                            yield SampleInput(CudaArrayInterfaceAttribute(obj),
                                              ref=RuntimeError("Can't pin tensor constructed from __cuda_array_interface__"),
                                              kwargs=kwargs, extra=extra)
                    else:
                        yield SampleInput(CudaArrayInterfaceAttribute(obj), ref=r, kwargs=kwargs, extra=extra)
                        yield SampleInput(CudaArrayInterfaceProperty(obj), ref=r, kwargs=kwargs, extra=extra)

    if dtype == torch.float32:
        # samples with bad inputs to tensor creation ops

        # arbitrary object
        yield SampleInput(object(), ref=RuntimeError('Could not infer dtype'))

        # python data with heterogeneous types
        yield SampleInput([0, 'torch'], ref=TypeError('invalid data type'))

        # python data with self-referential lists
        z = [0]
        z += [z]
        yield SampleInput(z, ref=TypeError('self-referential lists are incompatible'))
        yield SampleInput([[1, 2], z], ref=TypeError('self-referential lists are incompatible'))

        class BadCpuArrayInterface1:
            __array_interface__ = {}

        class BadCpuArrayInterface2:
            __array_interface__ = []

        class BadCpuArrayInterface3:
            @property
            def __array_interface__(self):
                raise IndexError('bogus index error')

        class BadCpuArrayInterface4:
            def __array__(self):
                return [1, 2]

        class BadCpuArrayInterface5:
            def __array__(self):
                raise IndexError('bogus index error')

        class BadCudaArrayInterface1:
            __cuda_array_interface__ = {}

        class BadCudaArrayInterface2:
            __cuda_array_interface__ = []

        class BadCudaArrayInterface3:
            @property
            def __cuda_array_interface__(self):
                raise IndexError('bogus index error')

        if device.type == 'cpu':
            # bad CPU Array Interface implementations
            yield SampleInput(BadCpuArrayInterface1(), ref=ValueError('Missing __array_interface__'))
            yield SampleInput(BadCpuArrayInterface2(), ref=ValueError('Invalid __array_interface__'))
            yield SampleInput(BadCpuArrayInterface3(), ref=IndexError('bogus index error'))
            yield SampleInput(BadCpuArrayInterface4(), ref=ValueError('object __array__ method not producing an array'))
            yield SampleInput(BadCpuArrayInterface5(), ref=IndexError('bogus index error'))
        elif device.type == 'cuda':
            # bad CUDA Array Interface implementations
            yield SampleInput(BadCudaArrayInterface1(), ref=ValueError('Missing __cuda_array_interface__'))
            yield SampleInput(BadCudaArrayInterface2(), ref=ValueError('Invalid __cuda_array_interface__'))
            yield SampleInput(BadCudaArrayInterface3(), ref=IndexError('bogus index error'))


tensor_op_db = [TensorCreationOpInfo('as_tensor',
                                     op=torch.as_tensor,
                                     sample_inputs_func=sample_inputs_tensor,
                                     supports_out=False),
                TensorCreationOpInfo('tensor',
                                     op=torch.tensor,
                                     sample_inputs_func=sample_inputs_tensor,
                                     supports_out=False),
                TensorCreationOpInfo('tensor_pinning',
                                     op=torch.tensor,
                                     sample_inputs_func=functools.partial(sample_inputs_tensor, pin_memory=True),
                                     supports_out=False),
                TensorCreationOpInfo('tensor_grad',
                                     op=torch.tensor,
                                     sample_inputs_func=sample_inputs_tensor,
                                     supports_autograd=True,
                                     supports_out=False),
                TensorCreationOpInfo('tensor_grad_pinning',
                                     op=torch.tensor,
                                     sample_inputs_func=functools.partial(sample_inputs_tensor, pin_memory=True),
                                     supports_autograd=True,
                                     supports_out=False)]


class TestTensorCreationOps(TestCase):

    def _check_shares_memory(self, tensor, other):
        """Check if two tensors share memory.
        """
        # Notice that tensors may share memory even when using
        # different devices: recall pin_memory option
        if tensor.layout == torch.strided:
            if tensor.storage().data_ptr() == other.storage().data_ptr():
                return True
            # TODO: check for strides
        return False

    def _apply_samples(self, device, dtype, op, test_func):
        samples = op.sample_inputs(device, dtype, requires_grad=op.supports_autograd)
        exceptions = []
        count = 0
        for sample in samples:
            count += 1
            ref = sample.ref
            args = sample.args if sample.input is sample.unspecified else (sample.input,) + sample.args
            assert sample.ref is not sample.unspecified
            if callable(ref):
                ref = ref(*args, **sample.kwargs)
            try:
                if isinstance(ref, Exception):
                    assert len(ref.args) == 1
                    try:
                        with self.assertRaisesRegex(type(ref), ref.args[0]):
                            op(*args, **sample.kwargs)
                    except Exception as exc:
                        assert ref == exc
                else:
                    result = op(*args, **sample.kwargs)
                    test_func(sample, result, ref)
            except Exception as exc:
                exceptions.append(f'{type(exc).__name__}:{exc} [{sample}]')
                if count > 10 and len(exceptions) == count:
                    # when all the first 10 samples have triggered
                    # failures, stop the test as it is unlikely that
                    # other samples do better
                    exceptions.append('Test stopped after excessive failures: some samples may be untested')
                    break
        if exceptions:
            msg = '\n  '.join(exceptions)
            raise AssertionError(f'{len(exceptions)} samples out of {count} triggered failures\n  {msg}')

    @suppress_warnings
    @onlyOnCPUAndCUDA
    @ops(tensor_op_db)
    def test_constructor(self, device, dtype, op):
        # torch.tensor, torch.as_tensor

        def test(sample, result, expected):

            self.assertEqual(result.device, expected.device)
            self.assertEqual(result.dtype, expected.dtype)
            self.assertEqual(result.layout, expected.layout)
            self.assertEqual(result.size(), expected.size())

            if not sample.extra.get('skip_equal_test'):
                self.assertEqual(result, expected)   # approximate equality
                assert (result == expected).all()    # exact equality

            if 'may_share_memory' in sample.extra:
                if sample.extra['may_share_memory']:
                    self.assertTrue(self._check_shares_memory(expected, result))
                else:
                    self.assertFalse(self._check_shares_memory(expected, result))

            if sample.extra.get('enable_pin_test'):
                self.assertTrue(result.is_pinned)


        self._apply_samples(device, dtype, op, test)


instantiate_device_type_tests(TestTensorCreationOps, globals())

if __name__ == '__main__':
    run_tests()
