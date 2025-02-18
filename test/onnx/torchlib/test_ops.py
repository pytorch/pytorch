# Owner(s): ["module: onnx"]
"""Test op correctness by comparing with PyTorch results.

Usage:

    pytest test_ops.py

    To run tests on a specific operator (e.g. torch.ceil):

    pytest test_ops.py -k ceil

    To run tests on a nn operator (e.g. nn.functional.scaled_dot_product_attention):

    pytest test_ops.py -k nn_functional_scaled_dot_product_attention

## Environment variables

1. Set environment variable `CATCH_ORT_SEGFAULT=1` to catch segmentation faults
in onnxruntime by running the inference sessions in a separate process.

2. Set `CREATE_REPRODUCTION_REPORT=1` to create markdown files for reproduction of
errors.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence, Tuple, TYPE_CHECKING

import error_reproduction
import numpy as np

import onnx
import onnxruntime as ort
import onnxscript
import ops_test_common
import ops_test_data
import parameterized

import torch
from torch.testing._internal import common_device_type, common_utils
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    import unittest

    from torch.testing._internal.opinfo import core as opinfo_core

# All dtypes will be tested on the generated symbolic functions.
# complex64 will be flattened to float32.
TESTED_DTYPES = (
    torch.float16,
    torch.float32,
    # Uncomment below item when we really need testing it
    # torch.bfloat16,
    # torch.float64,
    torch.bool,
    # torch.int8,
    # torch.int16,
    torch.int32,
    torch.int64,
    # torch.uint8,
)
# NOTE: torch.complex32 is experimental in torch
COMPLEX_TYPES = (torch.complex64,)


def dtypes_except(*dtypes: torch.dtype) -> Sequence[torch.dtype]:
    """Returns all dtypes except the ones specified."""
    return tuple(dtype for dtype in TESTED_DTYPES if dtype not in dtypes)


def _should_skip_xfail_test_sample(
    op_name: str, sample, dtype: torch.dtype, device_type: str
) -> Tuple[Optional[str], Optional[str]]:
    """Returns a reason if a test sample should be skipped."""
    if op_name not in ops_test_data.OP_WITH_SKIPPED_XFAIL_SUBTESTS:
        return None, None
    for decorator_meta in ops_test_data.SKIP_XFAIL_SUBTESTS:
        # Linear search on ops_test_data.SKIP_XFAIL_SUBTESTS. That's fine because the list is small.
        if decorator_meta.op_name == op_name:
            assert decorator_meta.matcher is not None, "Matcher must be defined"
            if not decorator_meta.enabled_if:
                # Do not skip the test if the decorator meta is not enabled
                continue
            if decorator_meta.dtypes is not None and dtype not in decorator_meta.dtypes:
                # Not applicable for this dtype
                continue
            if (
                decorator_meta.device_type is not None
                and decorator_meta.device_type != device_type
            ):
                # Not applicable for this device_type
                continue
            if decorator_meta.matcher(sample):
                return decorator_meta.test_behavior, decorator_meta.reason
    return None, None


class TestFunctionValidity(common_utils.TestCase):
    @parameterized.parameterized.expand(
        [
            (info.op.name, info)
            for info in ops_test_data.TESTED_TORCHLIB_OPS
            if isinstance(info.op, onnxscript.OnnxFunction)
        ]
    )
    def test_script_function_passes_checker(
        self, _, torchlib_op_info: ops_test_data.TorchLibOpInfo
    ):
        function_proto = torchlib_op_info.op.to_function_proto()
        onnx.checker.check_function(function_proto)  # type: ignore[attr-defined]


def run_test_output_match(
    test_suite: unittest.TestCase,
    device: str,
    dtype: torch.dtype,
    op: opinfo_core.OpInfo,
    function_executor: Callable,
    tested_op_mapping: dict[
        str,
        ops_test_data.TorchLibOpInfo,
    ],
):
    """Base test method for testing each opset, used by instantiate_device_type_tests.

    Args:
        test_suite: The test class instance.
        device: The PyTorch device. instantiate_device_type_tests provides this.
        dtype: The PyTorch dtype. instantiate_device_type_tests provides this.
        op: The OpInfo instance. instantiate_device_type_tests provides this.
        function_executor: The function executor. This is a function that takes
            a function and its arguments and returns the output of the function.
        tested_op_mapping: The mapping of op name to the tested op.
    """
    samples = op.sample_inputs(
        device,
        dtype,
        requires_grad=False,
    )

    torchlib_op_info = tested_op_mapping[op.name]
    # Obtain the input_wrangler that manipulates the OpInfo inputs
    # to match the aten operator signature
    # An example is nn.functional.upsample_nearest2d, which has a different signature
    # than the aten operator upsample_nearest2d
    onnx_function = torchlib_op_info.op
    input_wrangler = torchlib_op_info.input_wrangler
    if (
        not ops_test_common.dtype_op_schema_compatible(dtype, onnx_function.op_schema)
        and dtype not in COMPLEX_TYPES
    ):
        test_suite.skipTest(
            f"dtype '{dtype}' is not supported by the op '{op.name}'. "
            f"Type constraints: {onnx_function.op_schema.type_constraints}"
        )

    # Obtain the tolerance for the op
    rtol, atol = torchlib_op_info.get_tolerance(dtype)
    for i, cpu_sample in enumerate(samples):
        inputs = (cpu_sample.input, *cpu_sample.args)
        # Provide the repr to subtest because tensors are not serializable in parallel test runs
        with test_suite.subTest(
            sample_num=i,
            inputs=repr(
                [
                    f"Tensor<{inp.shape}, dtype={inp.dtype}>"
                    if isinstance(inp, torch.Tensor)
                    else inp
                    for inp in inputs
                ]
            ),
            kwargs=repr(cpu_sample.kwargs),
        ):
            try:
                device_type = cpu_sample.args[0].device.type
            except (AttributeError, IndexError):
                device_type = "cpu"
            test_behavior, reason = _should_skip_xfail_test_sample(
                op.name, cpu_sample, dtype, device_type
            )

            with ops_test_common.normal_xfail_skip_test_behaviors(
                test_behavior, reason
            ):
                input_onnx = [
                    ops_test_common.convert_tensor_to_numpy(x) for x in inputs
                ]
                kwargs_onnx = ops_test_common.convert_kwargs_for_onnx(cpu_sample.kwargs)
                if input_wrangler:
                    input_onnx, kwargs_onnx = input_wrangler(input_onnx, kwargs_onnx)
                torch_output = op(*inputs, **cpu_sample.kwargs)

                if isinstance(torch_output, torch.Tensor) and torch.is_complex(
                    torch_output
                ):
                    torch_output = torch.view_as_real(torch_output.resolve_conj())

                reference_torch_outputs, _ = pytree.tree_flatten(torch_output)
                if (
                    op.name.startswith("split")
                    or op.name.startswith("chunk")
                    or op.name.startswith("unbind")
                    or op.name
                    in {
                        "atleast_1d_Sequence",
                        "atleast_2d_Sequence",
                        "atleast_3d_Sequence",
                    }
                ):
                    # Hack for handling split, chunk and unbind which relies on SplitToSequence op.
                    # Split returns a Sequence that should be treats as a single
                    # value. So we wrap it into a tuple.
                    # TODO(justinchuby): Find a more general solution
                    reference_torch_outputs = [reference_torch_outputs]

                test_name = test_suite.id()
                function_output, model_proto = function_executor(
                    test_name, reference_torch_outputs
                )(onnx_function, input_onnx, kwargs_onnx)
                # Finally we re-flatten everything
                # TODO: add pytree structure comparison.
                flattened_torch_outputs, _ = pytree.tree_flatten(torch_output)
                flattened_function_outputs, _ = pytree.tree_flatten(function_output)

                assert flattened_torch_outputs
                assert len(flattened_torch_outputs) == len(flattened_function_outputs)

                for j, (torch_output, function_output) in enumerate(
                    zip(flattened_torch_outputs, flattened_function_outputs)
                ):
                    actual = torch.tensor(function_output)
                    expected = (
                        torch_output
                        if isinstance(torch_output, torch.Tensor)
                        else torch.tensor(torch_output)
                    )

                    if (
                        op.name in ops_test_data.NONDETERMINISTIC_OPS
                        or j in ops_test_data.COMPARE_SHAPE_ONLY_OPS[op.name]
                    ):
                        # Check shape and dtype only for ops that are known to be
                        # nondeterministic
                        test_suite.assertEqual(actual.shape, expected.shape)
                        test_suite.assertEqual(actual.dtype, expected.dtype)
                        continue

                    # Use torch.testing as opposed to np.testing to ensure dtypes and shapes match
                    try:
                        torch.testing.assert_close(
                            actual,
                            expected,
                            rtol=rtol,
                            atol=atol,
                            equal_nan=True,
                            check_device=False,
                        )
                    except AssertionError as e:
                        if (
                            os.environ.get("CREATE_REPRODUCTION_REPORT") == "1"
                            and test_behavior is None
                        ):
                            error_reproduction.create_mismatch_report(
                                test_name,
                                i,
                                model_proto,
                                inputs,
                                cpu_sample.kwargs,
                                actual,
                                expected,
                                e,
                                __file__,
                            )
                        if len(flattened_torch_outputs) > 1:
                            raise AssertionError(f"Output {j} mismatch") from e
                        raise


class TestOutputConsistencyFullGraph(common_utils.TestCase):
    """Test output consistency between exported ONNX op run as a graph and PyTorch eager mode.

    This is a parameterized test suite.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        ort.set_seed(42)

    @ops_test_common.add_decorate_info(
        ops_test_data.OPS_DB,
        "TestOutputConsistencyFullGraph",
        "test_output_match_opinfo_",
        skip_or_xfails=ops_test_data.EXPECTED_SKIPS_OR_FAILS,
    )
    @common_device_type.ops(  # type: ignore[misc]
        [
            info
            for info in ops_test_data.OPS_DB
            if info.name in ops_test_data.TESTED_OPS
        ],
        allowed_dtypes=TESTED_DTYPES,
    )
    def test_output_match_opinfo_(
        self, device: str, dtype: torch.dtype, op: opinfo_core.OpInfo
    ):
        # Base test method for testing each op by running the full ONNX graph.
        run_test_output_match(
            self,
            device,
            dtype,
            op,
            ops_test_common.graph_executor,
            ops_test_data.TORCHLIB_OPINFO_MAPPING,
        )

    @ops_test_common.add_decorate_info(
        ops_test_data.OPS_DB,
        "TestOutputConsistencyFullGraph",
        "test_complex_output_match_opinfo_",
        skip_or_xfails=ops_test_data.EXPECTED_SKIPS_OR_FAILS,
    )
    @common_device_type.ops(  # type: ignore[misc]
        [
            info
            for info in ops_test_data.OPS_DB
            if info.name in ops_test_data.COMPLEX_FUNCTION_MAPPING
        ],
        allowed_dtypes=COMPLEX_TYPES,
    )
    def test_complex_output_match_opinfo_(
        self, device: str, dtype: torch.dtype, op: opinfo_core.OpInfo
    ):
        """Base test method for testing each op by running the full ONNX graph."""
        run_test_output_match(
            self,
            device,
            dtype,
            op,
            ops_test_common.graph_executor,
            ops_test_data.COMPLEX_FUNCTION_MAPPING,
        )


common_device_type.instantiate_device_type_tests(
    TestOutputConsistencyFullGraph, globals(), only_for=["cpu"]
)

if __name__ == "__main__":
    common_utils.run_tests()
