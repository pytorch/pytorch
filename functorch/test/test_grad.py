from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.nn.functional as F
from torch import Tensor
import functools
import itertools
import copy
import warnings
import unittest
from torch.testing._internal.common_device_type import instantiate_device_type_tests, \
    skipCUDAIfNoMagma
from torch.testing._internal.common_device_type import ops, onlyCPU
from torch.testing._internal.common_methods_invocations import op_db
from common_utils import parameterized, instantiate_parameterized_methods
import types
from functorch import grad
from functorch._src.eager_transforms import _as_tuple

# Version of autograd.grad that handles outputs that don't depend on inputs
def _autograd_grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True):
    result = [torch.zeros_like(inp) for inp in inputs]
    diff_argnums = tuple(i for i, inp in enumerate(inputs) if inp.requires_grad)
    inputs = tuple(inputs[i] for i in diff_argnums)
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        result = tuple((out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad)
        if len(result) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*result)
    if len(diff_outputs) == 0:
        return tuple(torch.zeros_like(inp) for inp in inputs)
    grad_inputs = torch.autograd.grad(diff_outputs, inputs, grad_outputs,
                                      retain_graph=retain_graph,
                                      create_graph=create_graph,
                                      allow_unused=True)
    grad_inputs = tuple(torch.zeros_like(inp) if gi is None else gi
                        for gi, inp in zip(grad_inputs, inputs))
    for idx, grad_inp in zip(diff_argnums, grad_inputs):
        result[idx] = grad_inp
    return tuple(result)


class TestGradOpInfo(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_op(self, device, dtype, op):
        print(op.name)
        op_skip = {
            '__getitem__',
            '__rpow__',
            'float_power',
            'index_put',
            'linalg.cholesky',
            'linalg.inv',
            'linalg.matrix_norm',
            'linalg.matrix_power',
            'linalg.norm',
            'nanquantile',
            'pow',
            # CUDA-only failures
            'einsum',
            'linalg.multi_dot',
            'quantile',
        }
        if op.name in op_skip:
            self.skipTest("Skipped; Expected failures")
            return

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        def is_inplace(variant):
            if hasattr(variant, "__wrapped__"):
                return variant.__wrapped__ is op.get_inplace()
            return variant is op.get_inplace()

        for sample in samples:
            # TODO: test in-place
            if is_inplace(op.get_op()):
                continue

            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            # TODO: make this work on multilpe outputs
            result = op(sample.input, *sample.args, **sample.kwargs)
            if not isinstance(result, torch.Tensor):
                continue

            diff_argnums = tuple(i for i, arg in enumerate(args)
                                 if isinstance(arg, Tensor) and
                                 torch.is_floating_point(arg))
            diff_args = tuple(args[i] for i in diff_argnums)

            def wrapped_fn(*args, **kwargs):
                result = op(*args, **kwargs)
                return result.sum()

            result = grad(wrapped_fn, diff_argnums)(*args, **kwargs)
            expected = _autograd_grad(_as_tuple(wrapped_fn(*args, **kwargs)), diff_args)
            self.assertEqual(result, expected)

only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestGradOpInfo, globals(), only_for=only_for)

if __name__ == '__main__':
    run_tests()
