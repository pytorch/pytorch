# Owner(s): ["module: unknown"]

import collections
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.autocast_test_lists import AutocastCPUTestLists

class TestAutocastCPU(TestCase):
    def setUp(self):
        super(TestAutocastCPU, self).setUp()
        self.autocast_lists = AutocastCPUTestLists(torch.device('cpu'))

    def tearDown(self):
        del self.autocast_lists
        super(TestAutocastCPU, self).tearDown()

    def _run_autocast_outofplace(self, op, args, run_as_type, out_type=None, module=torch, add_kwargs=None):
        # helper to cast args
        def cast(val, to_type):
            if isinstance(val, torch.Tensor):
                return val.to(to_type) if val.is_floating_point() else val
            elif isinstance(val, collections.abc.Iterable):
                return type(val)(cast(v, to_type) for v in val)
            else:
                return val

        if add_kwargs is None:
            add_kwargs = {}

        self.assertFalse(torch.is_autocast_cpu_enabled())
        with torch.cpu.amp.autocast():
            self.assertTrue(torch.is_autocast_cpu_enabled())
            out_type = out_type if out_type is not None else run_as_type
            output = output_method = None

            # Try module.* variant, if requested:
            if module is not None and hasattr(module, op):
                output = getattr(module, op)(*args, **add_kwargs)
                if isinstance(output, torch.Tensor):
                    self.assertTrue(out_type == output.dtype,
                                    "autocast for torch.{} produced {}, should produce {}"
                                    .format(op, output.dtype, out_type))
            # Try Tensor.* variant:
            if hasattr(torch.Tensor, op):
                output_method = getattr(args[0], op)(*args[1:], **add_kwargs)
                if isinstance(output_method, torch.Tensor):
                    self.assertTrue(out_type == output_method.dtype,
                                    "autocast for torch.{} produced {}, should produce torch.{}"
                                    .format(op, output_method.dtype, out_type))

            self.assertTrue((output is not None) or (output_method is not None),
                            "{} not found as an attribute on either Tensor or the requested module {}".format(
                            op, module))

            # Accounts for ops that return Tensors, iterables, and other non-Tensors.
            # For example, lstm_cell returns a tuple and equal returns bool.
            def compare(first, second):
                if isinstance(first, torch.Tensor):
                    return torch.equal(first, second)
                elif isinstance(first, collections.abc.Iterable):
                    return all(compare(f, s) for f, s in zip(first, second))
                else:
                    return first == second

            # If both torch.* and Tensor.* variants were found, check outputs are identical
            if (output is not None) and (output_method is not None):
                self.assertTrue(type(output) == type(output_method))
                comparison = compare(output, output_method)
                self.assertTrue(comparison, "torch.{0} result did not match Tensor.{0} result".format(op))

            # Compare numerics to Python-side "autocasting" that (we expect) does the same thing
            # as the C++-side autocasting, and should be bitwise accurate.
            output_to_compare = output if output is not None else output_method
            with torch.cpu.amp.autocast(enabled=False):
                self.assertFalse(torch.is_autocast_cpu_enabled())

                if module is not None and hasattr(module, op):
                    control = getattr(module, op)(*cast(args, run_as_type), **add_kwargs)
                else:
                    control = getattr(args[0].to(run_as_type), op)(*cast(args[1:], run_as_type), **add_kwargs)
                self.assertTrue(type(output_to_compare) == type(control))
                comparison = compare(output_to_compare, control)
                self.assertTrue(comparison, "torch.{} result did not match control".format(op))
            self.assertTrue(torch.is_autocast_cpu_enabled())
        self.assertFalse(torch.is_autocast_cpu_enabled())

    def args_maybe_kwargs(self, op_with_args):
        if len(op_with_args) == 2:
            return op_with_args[0], op_with_args[1], {}
        else:
            return op_with_args[0], op_with_args[1], op_with_args[2]

    def test_autocast_torch_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(op, args, torch.float32, out_type=out_type)

    def test_autocast_methods_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.methods_expect_builtin_promote:
            self._run_autocast_outofplace(op, args, torch.float32, module=None, out_type=out_type)

    def test_autocast_torch_bf16(self):
        for op_with_args in self.autocast_lists.torch_bf16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.bfloat16, add_kwargs=maybe_kwargs)

    def test_autocast_nn_bf16(self):
        for op, args in self.autocast_lists.nn_bf16:
            self._run_autocast_outofplace(op, args, torch.bfloat16, module=torch._C._nn)

    def test_autocast_torch_fp32(self):
        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.float32, add_kwargs=maybe_kwargs)

    def test_autocast_nn_fp32(self):
        for op_with_args in self.autocast_lists.nn_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.float32, module=torch._C._nn, add_kwargs=maybe_kwargs)

    def test_autocast_torch_need_autocast_promote(self):
        for op, args in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(op, args, torch.float32)

class TestTorchAutocast(TestCase):
    def test_autocast_fast_dtype(self):
        gpu_fast_dtype = torch.get_autocast_gpu_dtype()
        cpu_fast_dtype = torch.get_autocast_cpu_dtype()
        self.assertEqual(gpu_fast_dtype, torch.half)
        self.assertEqual(cpu_fast_dtype, torch.bfloat16)


if __name__ == '__main__':
    run_tests()
