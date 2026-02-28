# Owner(s): ["module: PrivateUse1"]
"""
OpInfo-based operator tests for OpenReg backend.

This file demonstrates how third-party backends using PrivateUse1 can leverage
PyTorch's OpInfo testing infrastructure to validate operator implementations.

Key features demonstrated:
- Using @ops decorator to run tests across multiple operators
- Using skip() and skipOps() to handle known issues
- Testing non-contiguous tensor handling
- Testing out= argument semantics

"""

from collections.abc import Sequence

import torch
import torch_openreg  # noqa: F401
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db, skip, skipOps
from torch.testing._internal.common_utils import (
    first_sample,
    noncontiguous_like,
    run_tests,
    TestCase,
)


# supported operators for OpenReg - expand as more are implemented
OPENREG_OPS = {"add", "mul", "empty", "zeros", "ones"}

# filter op_db to only include supported operators
openreg_ops = [op for op in op_db if op.name in OPENREG_OPS]


# skips
noncontiguous_skips = {
    skip("empty"),
}


class TestOpInfo(TestCase):
    """OpInfo-based operator tests for the OpenReg backend.

    These tests validate that operators work correctly on the OpenReg device,
    including proper handling of non-contiguous inputs and out= arguments.
    """

    @ops(openreg_ops, allowed_dtypes=(torch.float32, torch.float16))
    def test_op_basic_functionality(self, device, dtype, op):
        device_type = torch.device(device).type

        sample = first_sample(self, op.sample_inputs(device, dtype))
        result = op(sample.input, *sample.args, **sample.kwargs)

        if isinstance(result, torch.Tensor):
            self.assertEqual(result.device.type, device_type)
        elif isinstance(result, (tuple, list)):
            for r in result:
                if isinstance(r, torch.Tensor):
                    self.assertEqual(r.device.type, device_type)

    @ops(
        [op for op in openreg_ops if op.supports_out],
        allowed_dtypes=(torch.float32,),
    )
    def test_out_argument(self, device, dtype, op):
        device_type = torch.device(device).type

        sample = first_sample(self, op.sample_inputs(device, dtype))
        expected = op(sample.input, *sample.args, **sample.kwargs)

        if isinstance(expected, torch.Tensor):
            out = torch.empty_like(expected)
            original_data_ptr = out.data_ptr()
            result = op(sample.input, *sample.args, **sample.kwargs, out=out)

            self.assertIs(result, out)
            self.assertEqual(out.device.type, device_type)
            self.assertEqual(out.data_ptr(), original_data_ptr)
            self.assertEqual(out, expected)

    @ops(openreg_ops, allowed_dtypes=(torch.float32,))
    @skipOps("TestOpInfo", "test_noncontiguous_samples", noncontiguous_skips)
    def test_noncontiguous_samples(self, device, dtype, op):
        if op.is_factory_function:
            self.skipTest(f"{op.name} is a factory function")

        test_grad = dtype in op.supported_backward_dtypes(torch.device(device).type)
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=test_grad)

        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = (
                sample_input.input,
                sample_input.args,
                sample_input.kwargs,
            )
            noncontig_sample = sample_input.noncontiguous()
            n_inp, n_args, n_kwargs = (
                noncontig_sample.input,
                noncontig_sample.args,
                noncontig_sample.kwargs,
            )

            # validates forward pass
            expected = op(t_inp, *t_args, **t_kwargs)
            actual = op(n_inp, *n_args, **n_kwargs)
            self.assertEqual(actual, expected)

            # validate backward pass
            if not test_grad:
                continue

            expected = sample_input.output_process_fn_grad(expected)
            actual = sample_input.output_process_fn_grad(actual)

            if isinstance(expected, torch.Tensor):
                grad_for_expected = torch.randn_like(expected)
                grad_for_actual = noncontiguous_like(grad_for_expected)
            elif isinstance(expected, Sequence):
                # filter output elements that do not require grad
                expected = [
                    t
                    for t in expected
                    if isinstance(t, torch.Tensor) and t.requires_grad
                ]
                actual = [
                    n for n in actual if isinstance(n, torch.Tensor) and n.requires_grad
                ]
                grad_for_expected = [torch.randn_like(t) for t in expected]
                grad_for_actual = [noncontiguous_like(n) for n in grad_for_expected]
            else:
                continue

            t_inputs = (
                (t_inp,) + t_args
                if isinstance(t_inp, torch.Tensor)
                else tuple(t_inp) + t_args
            )
            n_inputs = (
                (n_inp,) + n_args
                if isinstance(n_inp, torch.Tensor)
                else tuple(n_inp) + n_args
            )

            # filter the elements that are tensors that require grad
            t_input_tensors = [
                t for t in t_inputs if isinstance(t, torch.Tensor) and t.requires_grad
            ]
            n_input_tensors = [
                n for n in n_inputs if isinstance(n, torch.Tensor) and n.requires_grad
            ]

            self.assertEqual(len(t_input_tensors), len(n_input_tensors))

            t_grads = torch.autograd.grad(
                expected, t_input_tensors, grad_for_expected, allow_unused=True
            )
            n_grads = torch.autograd.grad(
                actual, n_input_tensors, grad_for_actual, allow_unused=True
            )

            msg = "Got different gradients for contiguous / non-contiguous inputs wrt input {}."
            for i, (t, n) in enumerate(zip(t_grads, n_grads)):
                self.assertEqual(t, n, msg=msg.format(i))


# instantiation
instantiate_device_type_tests(TestOpInfo, globals(), only_for=("openreg",))


if __name__ == "__main__":
    run_tests()
