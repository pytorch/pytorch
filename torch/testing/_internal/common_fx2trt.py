import unittest
from typing import Callable, List, Tuple

import torch
import torch.fx
import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
from torch.fx.experimental.fx2trt import (
    TRTInterpreter,
    InputTensorSpec,
    TRTModule,
)
from torch.testing._internal.common_utils import TestCase
from torch.fx.experimental.normalize import NormalizeArgs
from torch.fx.passes import shape_prop


def fetch_attr(mod, target):
    """
    Fetch an attribute from the ``Module`` hierarchy of ``mod.module``.

    Args:
        target (str): The fully-qualfiied name of the attribute to fetch

    Return:
        Any: The value of the attribute.
    """
    target_atoms = target.split(".")
    attr_itr = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TRTTestCase(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(3)

    def run_test(self, mod, inputs, expected_ops, unexpected_ops, interpreter, rtol, atol):
        with torch.no_grad():
            cuda_inputs = []
            for i in inputs:
                cuda_inputs.append(i.cuda())

            mod.eval()
            if len(expected_ops):
                self.assert_has_op(mod, expected_ops)
            if unexpected_ops:
                self.assert_unexpected_op(mod, unexpected_ops)

            interpreter_result = interpreter.run(fp16_mode=False)
            trt_mod = TRTModule(
                interpreter_result.engine,
                interpreter_result.input_names,
                interpreter_result.output_names,
            )

            ref_outputs = mod(*inputs)
            outputs = trt_mod(*cuda_inputs)

            if isinstance(outputs, torch.Tensor):
                ref_outputs = [ref_outputs]
                outputs = [outputs]

            for out, ref in zip(outputs, ref_outputs):
                torch.testing.assert_allclose(out.cpu(), ref, rtol=rtol, atol=atol)

    def run_test_custom_compare_results(
        self,
        mod,
        inputs,
        expected_ops,
        interpreter,
        comparators: List[Tuple[Callable, List]],
        fp16_mode=False,
    ):
        """
        Runs the test and compares the result using the provided comparators.
        The size of comparators must be equal to the number of outputs from 'mod'.

        mod          - a model to run.
        inputs       - a list of the model inputs.
        expected ops - a list of ops that should be verified.
        interpreter  - used for converting the model to TRT.
        comparators  - a list of (func, args) pairs corresponding to each of
                       the module outputs. usage: func(x, y, *args)

        """
        with torch.no_grad():
            cuda_inputs = []
            for i in inputs:
                cuda_inputs.append(i.cuda())

            mod.eval()
            if len(expected_ops):
                self.assert_has_op(mod, expected_ops)

            interpreter_result = interpreter.run(fp16_mode=fp16_mode)
            trt_mod = TRTModule(
                interpreter_result.engine,
                interpreter_result.input_names,
                interpreter_result.output_names,
            )
            res_trt = trt_mod(*cuda_inputs).cpu()
            res_cpu = mod(*inputs)
            assert len(res_trt) == len(res_cpu)
            assert len(res_cpu) == len(comparators)
            for output_trt, output_cpu, comparator in zip(
                res_trt, res_cpu, comparators
            ):
                comp_func = comparator[0]
                args = comparator[1]
                self.assertTrue(comp_func(output_trt, output_cpu, *args))

    def run_test_with_error(self, mod, inputs, interpreter, expect_error):
        with self.assertRaises(expect_error):
            with torch.no_grad():
                cuda_inputs = []
                for i in inputs:
                    cuda_inputs.append(i.cuda())

                mod.eval()
                interpreter.run(fp16_mode=False)

    def assert_has_op(self, mod, ops):
        ops_in_mod = set()

        for node in mod.graph.nodes:
            if node.op == "call_module":
                ops_in_mod.add(type(fetch_attr(mod, node.target)))
            elif node.op in {"call_function", "call_method"}:
                ops_in_mod.add(node.target)

        self.assertTrue(
            ops_in_mod >= ops, f"expected ops {ops}, actuall ops {ops_in_mod}"
        )

    def assert_unexpected_op(self, mod, ops):
        for node in mod.graph.nodes:
            if (node.op == "call_module"):
                if type(fetch_attr(mod, node.target)) in ops:
                    return False
            elif node.op in {"call_function", "call_method"}:
                if node.target in ops:
                    return False
        return True


class VanillaTestCase(TRTTestCase):
    def run_test(self, mod, inputs, expected_ops, rtol=1e-05, atol=1e-06):
        mod = torch.fx.symbolic_trace(mod)
        shape_prop.ShapeProp(mod).propagate(*inputs)
        mod = NormalizeArgs(mod).transform()
        interp = TRTInterpreter(mod, InputTensorSpec.from_tensors(inputs))
        super().run_test(mod, inputs, expected_ops, None, interp, rtol, atol)

    def run_test_custom_compare_results(
        self,
        mod,
        inputs,
        expected_ops,
        interpreter,
        comparators: List[Tuple[Callable, List]],
        fp16_mode=False,
    ):
        # interpreter is ignored, we do not need this for Vanilla tests
        # Note this is different from internal version, we need to fix the test case
        # after we refactor the internal callsites to use this file
        mod = torch.fx.symbolic_trace(mod)
        shape_prop.ShapeProp(mod).propagate(*inputs)
        mod = NormalizeArgs(mod).transform()
        interp = TRTInterpreter(mod, InputTensorSpec.from_tensors(inputs))
        super().run_test_custom_compare_results(
            mod, inputs, expected_ops, interp, comparators, fp16_mode=fp16_mode
        )


class AccTestCase(TRTTestCase):
    def run_test(
        self,
        mod,
        inputs,
        expected_ops,
        unexpected_ops=None,
        apply_passes=None,
        test_explicit_batch_dim=True,
        test_implicit_batch_dim=True,
        rtol=1e-03,
        atol=1e-03,
    ):
        mod.eval()
        mod = acc_tracer.trace(mod, inputs)

        if apply_passes is not None:
            for p in apply_passes:
                mod = p(mod)

        if test_implicit_batch_dim:
            interp = TRTInterpreter(mod, InputTensorSpec.from_tensors(inputs))
            super().run_test(mod, inputs, expected_ops, unexpected_ops, interp, rtol, atol)

        if test_explicit_batch_dim:
            interp = TRTInterpreter(
                mod, InputTensorSpec.from_tensors(inputs), explicit_batch_dimension=True
            )
            super().run_test(mod, inputs, expected_ops, unexpected_ops, interp, rtol, atol)

    def run_test_with_assert_error(
        self,
        mod,
        inputs,
        expect_error,
        test_explicit_batch_dim=True,
        test_implicit_batch_dim=True,
    ):
        mod.eval()
        mod = acc_tracer.trace(mod, inputs)

        if test_implicit_batch_dim:
            interp = TRTInterpreter(mod, InputTensorSpec.from_tensors(inputs))
            super().run_test_with_error(mod, inputs, interp, expect_error)

        if test_explicit_batch_dim:
            interp = TRTInterpreter(
                mod, InputTensorSpec.from_tensors(inputs), explicit_batch_dimension=True
            )
            super().run_test_with_error(mod, inputs, interp, expect_error)

    def run_test_with_dynamic_shape(
        self,
        mod,
        input_specs,
        expected_ops,
        unexpected_ops=None,
        rtol=1e-03,
        atol=1e-03,
    ):
        mod.eval()
        inputs = InputTensorSpec.create_inputs_from_specs(input_specs)
        mod = acc_tracer.trace(mod, inputs)
        interp = TRTInterpreter(mod, input_specs, explicit_batch_dimension=True)
        super().run_test(mod, inputs, expected_ops, unexpected_ops, interp, rtol, atol)
