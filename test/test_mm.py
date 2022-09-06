"""
Test bmm operation on tensors with various layouts.
"""

import torch
from functools import wraps
import unittest

from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, precisionOverride, toleranceOverride, tol)
from torch.testing._internal.common_methods_invocations import \
    (op_db, SampleInput)
from torch.testing._internal.common_utils import (TestCase, _TestParametrizer, run_tests)

bmm_ops = [op for op in op_db if op.name == 'bmm']


def _get_nnz(input):
    if input.layout == torch.strided:
        return input.numel()
    if input.layout == torch.sparse_coo:
        return input._values().numel()
    return input.values().numel()


def get_exception_mm(op_name, input, *args, **kwargs):
    """Return exception class and regex message if inputs are not
    supported, otherwise return (None, None).
    """
    if op_name == 'bmm':
        device, dtype = input.device, input.dtype
        other = args[0]
        # Do we want to test ops against inputs with mixed devices and dtypes?
        assert other.device == device and other.dtype == dtype
        this_layout, this_shape, this_nnz = input.layout, input.shape, _get_nnz(input)
        other_layout, other_shape, other_nnz = other.layout, other.shape, _get_nnz(other)

        # All the following if-blocks should be considered as PyTorch
        # issues of an unimplemented support for the given input
        # conditions. Eventually, all these if-blocks should be
        # eliminated.

        if this_layout == torch.strided:
            if other_layout == torch.sparse_coo:
                return RuntimeError, "bmm_sparse: Tensor 'mat2' must be dense"
            if other_layout == torch.sparse_csr:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if device.type == 'cuda':
                    return RuntimeError, "torch.baddbmm: Expected self to be strided, but got layout SparseCsr"
            if other_layout == torch.sparse_csc:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if device.type == 'cuda':
                    return RuntimeError, "torch.baddbmm: Expected self to be strided, but got layout SparseCsc"

        if this_layout == torch.sparse_coo:
            if other_layout == torch.strided:
                if dtype == torch.bfloat16 and this_nnz > 0 and device.type == 'cpu':
                    return RuntimeError, "\"bmm_sparse_dense\" not implemented for 'BFloat16'"
                if dtype == torch.bfloat16 and this_nnz > 0 and other_nnz > 0 and device.type == 'cuda':
                    return RuntimeError, "\"bmm_sparse_cuda\" not implemented for 'BFloat16'"
                if dtype == torch.float16 and this_nnz > 0 and other_nnz > 0 and device.type == 'cuda':
                    return RuntimeError, "\"bmm_sparse_cuda\" not implemented for 'Half'"
                if dtype == torch.complex128 and device.type == 'cuda' and this_nnz > 0 and other_nnz > 0:
                    return RuntimeError, "Tensor types must be either float32 or float64"
                if dtype in {torch.complex64, torch.complex128} and device.type == 'cuda' and this_nnz > 0 and other_nnz > 0:
                    return RuntimeError, "Tensor types must be either float32 or float64"
            if other_layout == torch.sparse_coo:
                return RuntimeError, "bmm_sparse: Tensor 'mat2' must be dense"
            if other_layout == torch.sparse_csr:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if device.type == 'cuda':
                    return RuntimeError, "torch.baddbmm: Expected self to be strided, but got layout SparseCsr"
            if other_layout == torch.sparse_csc:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if device.type == 'cuda':
                    return RuntimeError, "torch.baddbmm: Expected self to be strided, but got layout SparseCsc"

        if this_layout == torch.sparse_csr:
            if other_layout == torch.strided:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if(device.type == 'cuda' and dtype == torch.bfloat16
                   and torch.cuda.get_device_capability() < (8, 0) and this_nnz > 0):
                    return (RuntimeError,
                            "Sparse operations with CUDA tensors of BFloat16 type are not supported"
                            " on GPUs with compute capability < 8.0")
                if device.type == 'cuda' and this_nnz > 0:
                    return (RuntimeError,
                            'false INTERNAL ASSERT FAILED at "../aten/src/ATen/cuda/CUDASparseDescriptors.cpp":187, please report'
                            ' a bug to PyTorch. Support for batched CSR indices and values is not implemented.')
            if other_layout == torch.sparse_coo:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if device.type == 'cuda':
                    return RuntimeError, "torch.baddbmm: Expected self to be strided, but got layout Sparse"
            if other_layout == torch.sparse_csr:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if device.type == 'cuda':
                    return RuntimeError, "torch.baddbmm: Expected self to be strided, but got layout SparseCsr"
            if other_layout == torch.sparse_csc:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if device.type == 'cuda':
                    return RuntimeError, "torch.baddbmm: Expected self to be strided, but got layout SparseCsc"

        if this_layout == torch.sparse_csc:
            if other_layout == torch.strided:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if device.type == 'cuda' and this_nnz > 0:
                    return RuntimeError, 'addmm: computation on CUDA is not implemented for Strided [+] SparseCsc @ Strided'
            if other_layout == torch.sparse_coo:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if device.type == 'cuda':
                    return RuntimeError, "torch.baddbmm: Expected self to be strided, but got layout Sparse"
            if other_layout == torch.sparse_csr:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if device.type == 'cuda':
                    return RuntimeError, "torch.baddbmm: Expected self to be strided, but got layout Sparse"
            if other_layout == torch.sparse_csc:
                if device.type == 'cpu':
                    return NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'SparseCsrCPU' backend"
                if device.type == 'cuda':
                    return RuntimeError, "torch.baddbmm: Expected self to be strided, but got layout SparseCsc"

        return None, None
    else:
        raise NotImplementedError(f'get_exception_mm not implemented for {op_name} yet')


get_exception_functions = dict(
    bmm=lambda *args, **kwargs: get_exception_mm('bmm', *args, **kwargs),
)


def reference_mm(op_name, input, *args, **kwargs):
    """Reference implementation of op.
    """
    if op_name == 'bmm':
        this = input.to_dense()
        other = args[0].to_dense()
        assert len(this.shape) == len(other.shape) == 3, (this.shape, other.shape)
        batches = torch.zeros(input.shape[:2] + other.shape[2:], dtype=input.dtype, device=input.device)
        if input.dtype == torch.bfloat16:
            # workaround RuntimeError: "addmm_sparse_cuda" not implemented for 'BFloat16'
            return reference_mm(op_name, input.to(torch.float32), other.to(torch.float32), **kwargs).to(torch.bfloat16)
        if input.dtype == torch.float16:
            # workaround RuntimeError: "addmm_sparse_cuda" not implemented for 'Half'
            return reference_mm(op_name, input.to(torch.float32), other.to(torch.float32), **kwargs).to(torch.float16)
        for i in range(input.shape[0]):
            torch.mm(input[i], other[i], out=batches[i])
        return batches
    else:
        raise NotImplementedError(f'reference_mm not implemented for {op_name} yet')


reference_functions = dict(
    bmm=lambda *args, **kwargs: reference_mm('bmm', *args, **kwargs),
)


class mm_layouts(_TestParametrizer):
    """Decorator class for parametrization of test function with an input
    layout argument and an extra argument of sample inputs generator.
    The sample_inputs generator provides samples with all supported
    layouts for mm arguments.
    """

    def _parametrize_test(self, test, generic_cls, device_cls):

        # TODO: add sparse_bsr and sparse_bsc to this list
        layouts = (torch.strided, torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)

        @wraps(test)
        def wrap(self, layout, device, dtype, op):
            layout_name = str(layout).lstrip('torch.')
            if layout == torch.strided:
                # strided layouts are always supported
                sample_inputs_func = op.reference_inputs
            elif layout == torch.sparse_coo:
                if not op.supports_sparse:
                    raise unittest.SkipTest(f"{op.name} does not support inputs with {layout_name} layout")
                sample_inputs_func = op.reference_inputs_sparse_coo
            elif layout == torch.sparse_csr:
                if not op.supports_sparse_csr:
                    raise unittest.SkipTest(f"{op.name} does not support inputs with {layout_name} layout")
                sample_inputs_func = op.reference_inputs_sparse_csr
            elif layout == torch.sparse_csc:
                if not op.supports_sparse_csc:
                    raise unittest.SkipTest(f"{op.name} does not support inputs with {layout_name} layout")
                sample_inputs_func = op.reference_inputs_sparse_csc
            else:
                raise NotImplementedError(f'{layout}')

            def sample_inputs_generator():
                for sample_input in sample_inputs_func(device, dtype):
                    yield sample_input
                    if len(sample_input.args) == 1:  # bmm, mm, ...
                        other = sample_input.args[0]
                        for other_layout in layouts:
                            if other.layout != other_layout:
                                yield SampleInput(sample_input.input.clone(),
                                                  args=(torch.sparse._to_layout(other, other_layout),),
                                                  kwargs=sample_input.kwargs)
                    elif len(sample_input.args) == 2:  # baddmm, ...
                        pass  # TODO
                    else:
                        raise NotImplementedError(len(sample_input.args))

            test(self, layout, device, dtype, op, sample_inputs_generator())

        for layout in layouts:
            yield (wrap, str(layout).lstrip('torch.'), {'layout': layout})


class TestMM(TestCase):

    @mm_layouts()
    @ops(bmm_ops)
    @toleranceOverride({torch.float16: tol(atol=5, rtol=1e-2)})
    @precisionOverride({torch.complex64: 3e-3, torch.float32: 1e-3})
    def test_reference_and_support(self, layout, device, dtype, op, sample_inputs):
        """Compare op results against reference implementation. In the case of
        unsupported inputs (read: support not implemented yet), check
        that the expected exceptions are raised.
        """
        get_exception = get_exception_functions[op.name]
        ref_op = reference_functions[op.name]
        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
            exc_type, exc_message = get_exception(t_inp, *t_args, **t_kwargs)
            if exc_type is None:
                actual = op.op(t_inp, *t_args, **t_kwargs)
                expected = ref_op(t_inp, *t_args, **t_kwargs)
                self.assertEqual(actual.layout, torch.strided)
                if ((t_inp.layout == t_args[0].layout == torch.strided
                     and actual.dtype == torch.bfloat16
                     and actual.device.type == 'cpu')):
                    # Don't bother testing this case as the CPU
                    # implementation of bmm on bfloat16 tensors
                    # returns too inaccurate results TODO: revisit
                    # the corresponding CPU implementation.
                    continue
                self.assertEqual(actual, expected)
            else:
                with self.assertRaisesRegex(exc_type, exc_message):
                    op.op(t_inp, *t_args, **t_kwargs)

    @mm_layouts()
    @ops(bmm_ops)
    @precisionOverride({torch.complex64: 1e-3, torch.float32: 3e-3})
    def test_transpose_invariant(self, layout, device, dtype, op, sample_inputs):
        """Test the invariant (a @ b) == (b.T @ a.T).T
        """
        get_exception = get_exception_functions[op.name]

        def transpose(input):
            return torch.sparse._transpose_copy(input, -2, -1)

        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
            a, b = t_inp, t_args[0]

            aT, bT = transpose(a), transpose(b)
            exc_type1, exc_message1 = get_exception(a, b, **t_kwargs)
            exc_type2, exc_message2 = get_exception(bT, aT, **t_kwargs)
            if exc_type1 is None and exc_type2 is None:
                ab = op.op(a, b, **t_kwargs)
                abT = op.op(bT, aT, **t_kwargs)
                self.assertEqual(ab, transpose(abT))


instantiate_device_type_tests(TestMM, globals(), except_for='meta')

if __name__ == "__main__":
    run_tests()
