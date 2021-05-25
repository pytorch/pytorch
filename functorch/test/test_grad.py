from torch.testing._internal.common_utils import TestCase, run_tests, is_iterable_of_tensors
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
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from functorch import grad, vjp
from functorch._src.eager_transforms import _as_tuple

# Version of autograd.grad that handles outputs that don't depend on inputs
def _autograd_grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True):
    inputs, inputs_spec = tree_flatten(inputs)
    result = [torch.zeros_like(inp) for inp in inputs]
    diff_argnums = tuple(i for i, inp in enumerate(inputs) if inp.requires_grad)
    inputs = tuple(inputs[i] for i in diff_argnums)
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        something = [(out, go) for out, go in zip(outputs, grad_outputs)
                     if out.requires_grad]
        if len(result) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*something)
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
    return tree_unflatten(result, inputs_spec)


def diff_arg(arg):
    if is_iterable_of_tensors(arg):
        if all([a.requires_grad for a in arg]):
            return True
        if all([not a.requires_grad for a in arg]):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and arg.requires_grad


class TestGradOpInfo(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_grad(self, device, dtype, op):
        op_skip = {
            '__getitem__',
            '__rpow__',
            'linalg.cholesky',
            'linalg.inv',
            'linalg.matrix_norm',
            'linalg.matrix_power',
            'linalg.norm',
            'nanquantile',
            'quantile',
            'tensor_split',
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
                self.skipTest("Skipped! NYI: inplace-testing not supported.")
                continue

            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            diff_argnums = tuple(i for i, arg in enumerate(args) if diff_arg(arg))
            assert len(diff_argnums) > 0
            diff_args = tuple(args[i] for i in diff_argnums)

            def wrapped_fn(*args, **kwargs):
                result = op(*args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    result = sample.output_process_fn_grad(result)

                # Reduce into single value for grad
                if isinstance(result, torch.Tensor):
                    return result.sum()
                result = sum([res.sum() for res in result])
                return result

            result = grad(wrapped_fn, diff_argnums)(*args, **kwargs)
            expected = _autograd_grad(_as_tuple(wrapped_fn(*args, **kwargs)), diff_args)

            self.assertEqual(result, expected)

    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_vjp(self, device, dtype, op):
        op_skip = {
            '__getitem__',
            '__rpow__',
            'linalg.cholesky',
            'linalg.inv',
            'linalg.matrix_norm',
            'linalg.matrix_power',
            'linalg.norm',
            'nanquantile',
            'quantile',
            'tensor_split',
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
                self.skipTest("Skipped! NYI: inplace-testing not supported.")
                continue

            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            diff_argnums = tuple(i for i, arg in enumerate(args) if diff_arg(arg))
            assert len(diff_argnums) > 0
            diff_args = tuple(args[i] for i in diff_argnums)

            def processed_op(*args, **kwargs):
                result = op(*args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    result = sample.output_process_fn_grad(result)
                if isinstance(result, tuple):
                    # TODO: Remove the following hack for namedtuples
                    result = tuple(result)
                    result = tuple(r for r in result if torch.is_floating_point(r))
                return result

            def wrapped_fn(*diff_args):
                _args = list(args)
                for num, arg in zip(diff_argnums, diff_args):
                    _args[num] = arg
                return processed_op(*_args, **kwargs)

            result, vjp_fn = vjp(wrapped_fn, *diff_args)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)
            result_vjps = vjp_fn(cotangents)

            result = processed_op(*args, **kwargs)
            result = _as_tuple(result)
            expected_vjps = _autograd_grad(result, diff_args, _as_tuple(cotangents))

            self.assertEqual(result_vjps, expected_vjps)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestGradOpInfo, globals(), only_for=only_for)

if __name__ == '__main__':
    run_tests()
