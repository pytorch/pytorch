# Owner(s): ["module: onnx"]
"""Common utils for testing operators."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import multiprocessing
import os
import pprint
import sys
import unittest
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, Optional, TypeVar

import error_reproduction
import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state
import onnxscript
import onnxscript.evaluator
import pytest
from onnxscript import ir

import torch
from torch.onnx._internal.exporter import _building, _ir_passes, _tensors
from torch.testing._internal.opinfo import core as opinfo_core


T = TypeVar("T")


# Convenience tuples for creating dtype lists when skipping or xfailing tests

BOOL_TYPES = (torch.bool,)

INT_TYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)

FLOAT_TYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
)


TEST_OPSET_VERSION = 18
IS_MACOS = sys.platform.startswith("darwin")
IS_WINDOWS = os.name == "nt"


@dataclasses.dataclass
class DecorateMeta:
    """A dataclass for storing information about a test case to skip or xfail.

    Adapted from functorch: functorch/test/common_utils.py
    """

    op_name: str
    variant_name: str
    decorator: Callable[..., Any]
    dtypes: Optional[Collection[torch.dtype]]
    device_type: Optional[str]
    reason: str
    test_behavior: str
    matcher: Optional[Callable[[Any], bool]] = None
    enabled_if: bool = True
    # The test_class_name to apply the decorator to. If None, the decorator is
    # applied to all test classes.
    test_class_name: Optional[str] = None


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    dtypes: Optional[Collection[torch.dtype]] = None,
    device_type: Optional[str] = None,
    matcher: Optional[Callable[[Any], Any]] = None,
    enabled_if: bool = True,
    test_class_name: Optional[str] = None,
) -> DecorateMeta:
    """Expects an OpInfo test to fail.

    Args:
        op_name: The name of the operator.
        variant_name: Optional OpInfo variant_test_name.
        reason: The reason for the failure.
        dtypes: The dtypes to expect the failure.
        device_type: Device type. E.g. "cpu", "cuda".
        matcher: A function that matches the test sample input. It is used only when
            the xfail is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether the xfail is enabled.
        test_class_name: The test class name to apply the xfail to. If None, the
            xfail is applied to all test classes.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        dtypes=dtypes,
        device_type=device_type,
        matcher=matcher,
        reason=reason,
        enabled_if=enabled_if,
        test_class_name=test_class_name,
        test_behavior="xfail",
    )


def skip(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    dtypes: Optional[Collection[torch.dtype]] = None,
    device_type: Optional[str] = None,
    matcher: Optional[Callable[[Any], Any]] = None,
    enabled_if: bool = True,
    test_class_name: Optional[str] = None,
) -> DecorateMeta:
    """Skips an OpInfo test.

    Args:
        op_name: The name of the operator.
        variant_name: Optional OpInfo variant_test_name.
        reason: The reason for skipping.
        dtypes: The dtypes to skip.
        device_type: Device type. E.g. "cpu", "cuda".
        matcher: A function that matches the test sample input. It is used only when
            the skip is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether the skip is enabled.
        test_class_name: The test class name to apply the skip to. If None, the skip
            is applied to all test classes.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Skip: {reason}"),
        dtypes=dtypes,
        device_type=device_type,
        reason=reason,
        matcher=matcher,
        enabled_if=enabled_if,
        test_class_name=test_class_name,
        test_behavior="skip",
    )


def add_decorate_info(
    all_opinfos: Sequence[opinfo_core.OpInfo],
    test_class_name: str,
    base_test_name: str,
    skip_or_xfails: Iterable[DecorateMeta],
) -> Callable[[T], T]:
    """Decorates OpInfo tests with decorators based on the skip_or_xfails list."""
    ops_mapping = {(info.name, info.variant_test_name): info for info in all_opinfos}
    for decorate_meta in skip_or_xfails:
        opinfo = ops_mapping.get((decorate_meta.op_name, decorate_meta.variant_name))
        if opinfo is None and not decorate_meta.enabled_if:
            # If the OpInfo doesn't exist and it is not enabled, we skip the OpInfo
            # because it could be an OpInfo that is in torch-nightly but not older versions.
            continue
        assert opinfo is not None, (
            f"Couldn't find OpInfo for {decorate_meta}. Did you need to specify variant_name?"
        )
        decorators = list(opinfo.decorators)
        new_decorator = opinfo_core.DecorateInfo(
            decorate_meta.decorator,
            decorate_meta.test_class_name or test_class_name,
            base_test_name,
            dtypes=decorate_meta.dtypes,
            device_type=decorate_meta.device_type,
            active_if=decorate_meta.enabled_if,
        )
        decorators.append(new_decorator)
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn

    return wrapped


def duplicate_opinfo(
    opinfos: list[opinfo_core.OpInfo], name: str, new_names: tuple[str, ...]
):
    """Duplicate an opinfo in the opinfo database and give it a new name."""
    duplicated = []
    all_info_names = {opinfo.name for opinfo in opinfos}
    for opinfo in opinfos:
        if opinfo.name == name:
            for new_name in new_names:
                if new_name in all_info_names:
                    # NOTE: Avoid duplicating an opinfo that already exists in the database.
                    # New opinfos are expected to be added in torch-nightly.
                    warnings.warn(
                        f"OpInfo {new_name} already exists in the database.",
                        stacklevel=1,
                    )
                    continue
                new_opinfo = copy.deepcopy(opinfo)
                new_opinfo.name = new_name
                duplicated.append(new_opinfo)
    opinfos.extend(duplicated)


def duplicate_opinfo_for_prims(
    opinfos: list[opinfo_core.OpInfo], name: str, prims_name: str | None = None
):
    """Duplicate an opinfo in the opinfo database for a prims op.

    The function sets the new OpInfo to use the variation torch.ops.prims.
    The new OpInfo will have the name "prims_{prims_name}" where `prims_name` is the
    name of the prims op. If `prims_name` is None, it will be set to "prims_{name}".

    Args:
        opinfos: The list of opinfo_core.OpInfo to add the new opinfo to.
        name: The name of the opinfo to duplicate.
        prims_name: The name of the prims op. If None, it will be set to `name`.
    """
    if prims_name is None:
        prims_name = name
    # The name of the new OpInfo
    new_name = f"prims_{prims_name}"
    all_info_names = {opinfo.name for opinfo in opinfos}
    for opinfo in opinfos:
        if opinfo.name == name:
            if new_name in all_info_names:
                # NOTE: Avoid duplicating an opinfo that already exists in the database.
                warnings.warn(
                    f"OpInfo {new_name} already exists in the database.", stacklevel=1
                )
                continue
            new_opinfo = copy.deepcopy(opinfo)
            new_opinfo.name = new_name
            new_opinfo.op = getattr(torch.ops.prims, prims_name)
            opinfos.append(new_opinfo)  # noqa: B909
            return
    raise RuntimeError(f"OpInfo '{name}' not found in the database.")


TORCH_TYPE_TO_ONNX = {
    torch.bool: onnx.TensorProto.BOOL,
    torch.uint8: onnx.TensorProto.UINT8,
    torch.int8: onnx.TensorProto.INT8,
    torch.int16: onnx.TensorProto.INT16,
    torch.int32: onnx.TensorProto.INT32,
    torch.int64: onnx.TensorProto.INT64,
    torch.float16: onnx.TensorProto.FLOAT16,
    torch.float32: onnx.TensorProto.FLOAT,
    torch.float64: onnx.TensorProto.DOUBLE,
    torch.complex64: onnx.TensorProto.COMPLEX64,
    torch.complex128: onnx.TensorProto.COMPLEX128,
    torch.bfloat16: onnx.TensorProto.BFLOAT16,
}


def convert_tensor_to_numpy(input: Any) -> Any:
    if isinstance(input, torch.Tensor):
        if torch.is_complex(input):
            # from complex to real representation
            input = torch.view_as_real(input)
        return input.detach().cpu().numpy()
    if isinstance(input, complex):
        return torch.view_as_real(torch.tensor(input)).detach().cpu().numpy()
    if isinstance(input, list):
        if len(input) == 0:
            return np.array((), dtype=np.int64)
        if any(isinstance(x, torch.Tensor) for x in input):
            # The list can be Optional[Tensor], e.g. [None, Tensor, None] etc.
            return [convert_tensor_to_numpy(x) for x in input]
        if isinstance(input[0], bool):
            return np.array(input, dtype=np.bool_)

        # Just a sequence of numbers
        if isinstance(input[0], int):
            return np.array(input, dtype=np.int64)
        if isinstance(input[0], float):
            return np.array(input)

    return input


def convert_kwargs_for_onnx(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Converts kwargs to be compatible with ONNX Runtime."""
    new_kwargs = {}
    for key, value in kwargs.items():
        if key == "device":
            continue
        if key == "dtype":
            value = TORCH_TYPE_TO_ONNX[value]
        if isinstance(value, torch.Tensor):
            value = np.array(value.cpu())
        new_kwargs[key] = value
    return new_kwargs


class OrtAbortedError(RuntimeError):
    """ONNX Runtime Aborted."""


def _ort_session_run(serialized_model: bytes, ort_inputs: Mapping[str, Any]):
    """Run a model with ONNX Runtime."""

    # Disable all ORT optimizations
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        serialized_model, session_options, providers=("CPUExecutionProvider",)
    )
    return session.run(None, ort_inputs)


def _ort_session_run_return_dict(
    serialized_model: bytes, ort_inputs: Mapping[str, Any], return_dict
) -> None:
    """Run a model with ONNX Runtime and store the results in return_dict."""

    try:
        return_dict["results"] = _ort_session_run(serialized_model, ort_inputs)
        return_dict["error"] = None
    except Exception as e:  # pylint: disable=broad-except
        return_dict["results"] = None
        return_dict["error"] = e


def _safe_ort_session_run(serialized_model: bytes, ort_inputs: Mapping[str, Any]):
    """Run a model with ONNX Runtime in a separate process.

    Args:
        serialized_model: Serialized ONNX model proto.
        ort_inputs: Inputs to the model.

    Returns:
        The inference result.

    Raises:
        OrtAbortedError if the process did not execute successfully.
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process = multiprocessing.Process(
        target=_ort_session_run_return_dict,
        args=(serialized_model, ort_inputs, return_dict),
    )
    process.start()
    process.join()
    process.close()
    if not return_dict:
        raise OrtAbortedError
    if return_dict["error"] is not None:
        raise return_dict["error"]
    return return_dict["results"]


def _format_model_and_input_information(onnx_model, inputs):
    return (
        f"Inputs:\n{pprint.pformat(inputs)}\nModel:\n{onnx.printer.to_text(onnx_model)}"
    )


_TORCH_DTYPE_TO_ONNX_STRING = {
    torch.bool: "tensor(bool)",
    torch.uint8: "tensor(uint8)",
    torch.int8: "tensor(int8)",
    torch.int16: "tensor(int16)",
    torch.int32: "tensor(int32)",
    torch.int64: "tensor(int64)",
    torch.float16: "tensor(float16)",
    torch.float32: "tensor(float)",
    torch.float64: "tensor(double)",
    torch.complex64: "tensor(complex64)",
    torch.complex128: "tensor(complex128)",
    torch.bfloat16: "tensor(bfloat16)",
}

_TORCH_DTYPE_TO_ONNX: dict[torch.dtype, ir.DataType] = {
    torch.bfloat16: ir.DataType.BFLOAT16,
    torch.bool: ir.DataType.BOOL,
    torch.complex128: ir.DataType.COMPLEX128,
    torch.complex64: ir.DataType.COMPLEX64,
    torch.float16: ir.DataType.FLOAT16,
    torch.float32: ir.DataType.FLOAT,
    torch.float64: ir.DataType.DOUBLE,
    torch.float8_e4m3fn: ir.DataType.FLOAT8E4M3FN,
    torch.float8_e4m3fnuz: ir.DataType.FLOAT8E4M3FNUZ,
    torch.float8_e5m2: ir.DataType.FLOAT8E5M2,
    torch.float8_e5m2fnuz: ir.DataType.FLOAT8E5M2FNUZ,
    torch.int16: ir.DataType.INT16,
    torch.int32: ir.DataType.INT32,
    torch.int64: ir.DataType.INT64,
    torch.int8: ir.DataType.INT8,
    torch.uint8: ir.DataType.UINT8,
    torch.uint16: ir.DataType.UINT16,
    torch.uint32: ir.DataType.UINT32,
    torch.uint64: ir.DataType.UINT64,
}


def dtype_op_schema_compatible(dtype: torch.dtype, schema: onnx.defs.OpSchema) -> bool:
    """Checks if the dtype is compatible with the schema.

    When a dtype is "compatible" with the schema, it means we can use the dtype
    to create sample inputs by OpInfo to test the ONNX function and expect outputs to match.

    Args:
        dtype: The torch dtype used to create sample inputs by OpInfo.
        schema: The ONNX schema of the function.

    Returns:
        True if the dtype is compatible with the schema.
    """
    if not schema.inputs:
        # If there are no inputs, we can't check compatibility. Assume it is compatible.
        # e.g. aten_randn has only attributes.
        return True
    if schema.inputs[0].name not in {"self", "input"}:
        # If the name of the first input is not "self" or "input",
        # it is usually an input that is not of the same type as the output.
        # We assume support in this case.
        #
        # For example, `aten_ones(size: IntType, dtype: int = FLOAT.dtype)`
        # has the first input as `size`, which is an integer, but it can support
        # any dtype.
        return True

    # Otherwise we check the type constraints of the first input.
    # For example, when dtype=torch.float32, and the op being tested has the schema
    # ```
    # OpSchema(
    #     name='aten_abs',
    #     domain='pkg.onnxscript.torch_lib',
    #     since_version=1,
    #     doc='abs(Tensor self) -> Tensor',
    #     type_constraints=[OpSchema.TypeConstraintParam(type_param_str='TReal',
    # allowed_type_strs=['tensor(float)', 'tensor(int8)', 'tensor(int16)',
    # 'tensor(int32)', 'tensor(int64)', 'tensor(float16)', 'tensor(double)',
    # 'tensor(bfloat16)'], description='')],
    #     inputs=[OpSchema.FormalParameter(name='self', type_str='TReal',
    # description='', param_option=<FormalParameterOption.Single: 0>,
    # is_homogeneous=True, min_arity=1,
    # differentiation_category=<DifferentiationCategory.Unknown: 0>)],
    #     outputs=[OpSchema.FormalParameter(name='return_val',
    # type_str='TReal', description='',
    # param_option=<FormalParameterOption.Single: 0>, is_homogeneous=True,
    # min_arity=1, differentiation_category=<DifferentiationCategory.Unknown: 0>)],
    #     attributes={}
    # )
    # ```
    # we see the first input type is "TReal", corresponding to the type constraint
    # with allowed types ['tensor(float)', 'tensor(int8)', 'tensor(int16)',
    # 'tensor(int32)', 'tensor(int64)', 'tensor(float16)', 'tensor(double)',
    # 'tensor(bfloat16)'].
    # Since torch.float32 (tensor(float)) is in the allowed types, we return True.

    first_input_type_name = schema.inputs[0].type_str
    # Find the type constraint for the first input by matching the parameter name
    first_input_type_constraint = next(
        (
            x
            for x in schema.type_constraints
            if first_input_type_name in x.type_param_str
        ),
        None,
    )
    assert first_input_type_constraint is not None
    allowed_type_strs = first_input_type_constraint.allowed_type_strs
    # Here we consider seq(tensor(float)) compatible with tensor(float) as well
    return any(
        _TORCH_DTYPE_TO_ONNX_STRING[dtype] in type_str for type_str in allowed_type_strs
    )


def graph_executor(
    test_name: str,
    outputs: Sequence[Any],
    opset_version: int = TEST_OPSET_VERSION,
) -> Callable[[Callable[..., Any], tuple[Any], dict[str, Any]], None]:
    """Eagerly executes a function."""

    def _capture_graph_and_evaluate_torch_script_evaluator(
        function: Callable, args, kwargs
    ) -> tuple[Any, onnx.ModelProto]:
        """Captures the graph of a function and evaluates it using TorchScriptEvaluator."""

        # Initialize the ONNX graph
        graph = ir.Graph(
            (),
            (),
            nodes=(),
            opset_imports={"": opset_version, "pkg.torch.onnx": 1},
            name="main_graph",
        )
        opset = onnxscript.values.Opset("", opset_version)
        tracer = _building.OpRecorder(opset, {})
        ort_inputs = {}
        onnxscript_args: list[Any] = []
        onnxscript_kwargs = {}
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                input_name = f"input_{i}"
                input = _tensors.SymbolicTensor(
                    opset=opset,
                    name=input_name,
                    shape=ir.Shape(arg.shape),
                    type=ir.TensorType(_TORCH_DTYPE_TO_ONNX[torch.tensor(arg).dtype]),
                )
                graph.inputs.append(input)
                onnxscript_args.append(input)
                ort_inputs[input_name] = arg
            elif isinstance(arg, (list, tuple)):
                # str is also a sequence but we do not want to treat it as a tensor
                sequence_input = []
                for j, subarg in enumerate(arg):
                    if isinstance(subarg, np.ndarray):
                        input_name = f"input_{i}_{j}"
                        tensor = torch.tensor(subarg)
                        input = _tensors.SymbolicTensor(
                            opset=opset,
                            name=input_name,
                            shape=ir.Shape(tensor.shape),
                            type=ir.TensorType(_TORCH_DTYPE_TO_ONNX[tensor.dtype]),
                        )
                        graph.inputs.append(input)
                        sequence_input.append(input)
                        ort_inputs[input_name] = subarg
                    else:
                        # Include non-numpy inputs as-is
                        # For example, it could be a None value that we want to keep
                        sequence_input.append(subarg)
                onnxscript_args.append(sequence_input)
            else:
                onnxscript_args.append(arg)
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                input = _tensors.SymbolicTensor(
                    opset=opset,
                    name=key,
                    shape=ir.Shape(torch.tensor(value).shape),
                    type=ir.TensorType(_TORCH_DTYPE_TO_ONNX[torch.tensor(value).dtype]),
                )
                graph.inputs.append(input)
                ort_inputs[key] = value
                onnxscript_kwargs[key] = input
            else:
                onnxscript_kwargs[key] = value

        with onnxscript.evaluator.default_as(tracer):
            symbolic_outputs = function(*onnxscript_args, **onnxscript_kwargs)
        if not isinstance(symbolic_outputs, Sequence):
            symbolic_outputs = (symbolic_outputs,)

        # We need to set the size of the output tensors for the ONNX model to be valid
        for output, symbolic_output in zip(outputs, symbolic_outputs):
            if isinstance(output, Sequence):
                # Output is a sequence
                elem_dtype = _TORCH_DTYPE_TO_ONNX[output[0].dtype]
                symbolic_output.type = ir.SequenceType(ir.TensorType(elem_dtype))
                continue
            output = (
                output
                if isinstance(output, torch.Tensor)
                else torch.tensor(output, device="cpu")
            )
            symbolic_output.shape = ir.Shape(output.shape)
            symbolic_output.dtype = _TORCH_DTYPE_TO_ONNX[output.dtype]

        graph.outputs.extend(symbolic_outputs)
        graph.extend(tracer.nodes)
        onnx_model = ir.Model(graph, ir_version=10, producer_name="torch_test")
        for identifier, onnxscript_function in tracer.functions.items():
            if identifier in onnx_model.functions:
                continue
            if isinstance(onnxscript_function, ir.Function):
                ir_function = onnxscript_function
            else:
                # TODO: Get IR function directly when onnxscript is updated
                proto = onnxscript_function.to_function_proto()
                ir_function = ir.serde.deserialize_function(proto)
            onnx_model.functions[identifier] = ir_function
        _ir_passes.add_opset_imports(onnx_model)
        # Make sure the model is valid
        model_proto = ir.to_proto(onnx_model)
        try:
            onnx.checker.check_model(model_proto, full_check=True)
        except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as e:
            raise AssertionError(f"ONNX model is invalid. Model:\n{onnx_model}") from e
        model_proto = onnx.shape_inference.infer_shapes(model_proto, data_prop=True)
        try:
            if (
                os.environ.get("CATCH_ORT_SEGFAULT") == "1"
                or os.environ.get("CREATE_REPRODUCTION_REPORT") == "1"
            ):
                # Use an individual process to run ONNX Runtime to catch segfaults
                return _safe_ort_session_run(
                    model_proto.SerializeToString(), ort_inputs
                ), model_proto

            return _ort_session_run(
                model_proto.SerializeToString(), ort_inputs
            ), model_proto
        except (
            # pylint: disable=c-extension-no-member
            onnxruntime.capi.onnxruntime_pybind11_state.Fail,
            onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException,
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument,
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph,
            onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented,
            # pylint: enable=c-extension-no-member
        ) as e:
            if os.environ.get("CREATE_REPRODUCTION_REPORT") == "1":
                error_reproduction.create_reproduction_report(
                    test_name,
                    model_proto,
                    ort_inputs,
                    e,
                    "test/onnx/torchlib/test_ops.py",
                )
            raise RuntimeError(
                "ONNX Runtime failed to evaluate:\n"
                + _format_model_and_input_information(model_proto, ort_inputs)
            ) from e
        except OrtAbortedError as e:
            if os.environ.get("CREATE_REPRODUCTION_REPORT") == "1":
                # Save the model and inputs to a file for reproduction
                error_reproduction.create_reproduction_report(
                    test_name,
                    model_proto,
                    ort_inputs,
                    e,
                    "test/onnx/torchlib/test_ops.py",
                )
            raise OrtAbortedError(
                "ONNX Runtime aborted:\n"
                + _format_model_and_input_information(model_proto, ort_inputs)
            ) from e
        except Exception as e:
            if os.environ.get("CREATE_REPRODUCTION_REPORT") == "1":
                error_reproduction.create_reproduction_report(
                    test_name,
                    model_proto,
                    ort_inputs,
                    e,
                    "test/onnx/torchlib/test_ops.py",
                )
            raise

    return _capture_graph_and_evaluate_torch_script_evaluator


@contextlib.contextmanager
def normal_xfail_skip_test_behaviors(
    test_behavior: Optional[str] = None, reason: Optional[str] = None
):
    """This context manager is used to handle the different behaviors of xfail and skip.

    Args:
        test_behavior (optional[str]): From DecorateMeta name, can be 'skip', 'xfail', or None.
        reason (optional[str]): The reason for the failure or skip.

    Raises:
        e: Any exception raised by the test case if it's not an expected failure.
    """

    # We need to skip as soon as possible, as SegFault might also be a case.
    if test_behavior == "skip":
        pytest.skip(reason=reason)

    try:
        yield
    # We could use `except (AssertionError, RuntimeError, ...) as e:`, but it needs
    # to go over all test cases to find the right exception type.
    except Exception:  # pylint: disable=broad-exception-caught
        if test_behavior is None:
            raise
        if test_behavior == "xfail":
            pytest.xfail(reason=reason)
    else:
        if test_behavior == "xfail":
            pytest.fail("Test unexpectedly passed")
