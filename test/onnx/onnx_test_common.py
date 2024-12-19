# Owner(s): ["module: onnx"]

from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import logging
import os
import unittest
import warnings
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import numpy as np
import onnxruntime
import pytest
import pytorch_test_common

import torch
from torch import export as torch_export
from torch.onnx import _constants, verification
from torch.testing._internal import common_utils
from torch.testing._internal.opinfo import core as opinfo_core
from torch.types import Number


_NumericType = Union[Number, torch.Tensor, np.ndarray]
_ModelType = Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
_InputArgsType = Optional[
    Union[torch.Tensor, int, float, bool, Sequence[Any], Mapping[str, Any]]
]
_OutputsType = Sequence[_NumericType]

onnx_model_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    "repos",
    "onnx",
    "onnx",
    "backend",
    "test",
    "data",
)


pytorch_converted_dir = os.path.join(onnx_model_dir, "pytorch-converted")


pytorch_operator_dir = os.path.join(onnx_model_dir, "pytorch-operator")


def run_model_test(test_suite: _TestONNXRuntime, *args, **kwargs):
    options = verification.VerificationOptions()

    kwargs["opset_version"] = test_suite.opset_version
    kwargs["keep_initializers_as_inputs"] = test_suite.keep_initializers_as_inputs
    if hasattr(test_suite, "check_shape"):
        options.check_shape = test_suite.check_shape
    if hasattr(test_suite, "check_dtype"):
        options.check_dtype = test_suite.check_dtype

    names = {f.name for f in dataclasses.fields(options)}
    keywords_to_pop = []
    for k, v in kwargs.items():
        if k in names:
            setattr(options, k, v)
            keywords_to_pop.append(k)
    for k in keywords_to_pop:
        kwargs.pop(k)

    return verification.verify(*args, options=options, **kwargs)


def assert_dynamic_shapes(onnx_program: torch.onnx.ONNXProgram, dynamic_shapes: bool):
    """Assert whether the exported model has dynamic shapes or not.

    Args:
        onnx_program (torch.onnx.ONNXProgram): The output of torch.onnx.dynamo_export.
        dynamic_shapes (bool): Whether the exported model has dynamic shapes or not.
            When True, raises if graph inputs don't have at least one dynamic dimension
            When False, raises if graph inputs have at least one dynamic dimension.

    Raises:
        AssertionError: If the exported model has dynamic shapes and dynamic_shapes is False and vice-versa.
    """

    if dynamic_shapes is None:
        return

    model_proto = onnx_program.model_proto
    # Process graph inputs
    dynamic_inputs = []
    for inp in model_proto.graph.input:
        dynamic_inputs += [
            dim
            for dim in inp.type.tensor_type.shape.dim
            if dim.dim_value == 0 and dim.dim_param != ""
        ]
    assert dynamic_shapes == (
        len(dynamic_inputs) > 0
    ), "Dynamic shape check failed for graph inputs"


def parameterize_class_name(cls: Type, idx: int, input_dicts: Mapping[Any, Any]):
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    suffix = "_".join(f"{k}_{v}" for k, v in input_dicts.items())
    return f"{cls.__name__}_{suffix}"


class _TestONNXRuntime(pytorch_test_common.ExportTestCase):
    opset_version = _constants.ONNX_DEFAULT_OPSET
    keep_initializers_as_inputs = True  # For IR version 3 type export.
    is_script = False
    check_shape = True
    check_dtype = True

    def setUp(self):
        super().setUp()
        onnxruntime.set_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"
        self.is_script_test_enabled = True

    # The exported ONNX model may have less inputs than the pytorch model because of const folding.
    # This mostly happens in unit test, where we widely use torch.size or torch.shape.
    # So the output is only dependent on the input shape, not value.
    # remained_onnx_input_idx is used to indicate which pytorch model input idx is remained in ONNX model.
    def run_test(
        self,
        model,
        input_args,
        input_kwargs=None,
        rtol=1e-3,
        atol=1e-7,
        do_constant_folding=True,
        dynamic_axes=None,
        additional_test_inputs=None,
        input_names=None,
        output_names=None,
        fixed_batch_size=False,
        training=torch.onnx.TrainingMode.EVAL,
        remained_onnx_input_idx=None,
        verbose=False,
    ):
        def _run_test(m, remained_onnx_input_idx, flatten=True, ignore_none=True):
            return run_model_test(
                self,
                m,
                input_args=input_args,
                input_kwargs=input_kwargs,
                rtol=rtol,
                atol=atol,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                additional_test_inputs=additional_test_inputs,
                input_names=input_names,
                output_names=output_names,
                fixed_batch_size=fixed_batch_size,
                training=training,
                remained_onnx_input_idx=remained_onnx_input_idx,
                flatten=flatten,
                ignore_none=ignore_none,
                verbose=verbose,
            )

        if isinstance(remained_onnx_input_idx, dict):
            scripting_remained_onnx_input_idx = remained_onnx_input_idx["scripting"]
            tracing_remained_onnx_input_idx = remained_onnx_input_idx["tracing"]
        else:
            scripting_remained_onnx_input_idx = remained_onnx_input_idx
            tracing_remained_onnx_input_idx = remained_onnx_input_idx

        is_model_script = isinstance(
            model, (torch.jit.ScriptModule, torch.jit.ScriptFunction)
        )

        if self.is_script_test_enabled and self.is_script:
            script_model = model if is_model_script else torch.jit.script(model)
            _run_test(
                script_model,
                scripting_remained_onnx_input_idx,
                flatten=False,
                ignore_none=False,
            )
        if not is_model_script and not self.is_script:
            _run_test(model, tracing_remained_onnx_input_idx)

    def run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
        self,
        model: _ModelType,
        input_args: Sequence[_InputArgsType],
        *,
        input_kwargs: Optional[Mapping[str, _InputArgsType]] = None,
        rtol: Optional[float] = 1e-3,
        atol: Optional[float] = 1e-7,
        has_mutation: bool = False,
        additional_test_inputs: Optional[
            List[
                Union[
                    Tuple[Sequence[_InputArgsType], Mapping[str, _InputArgsType]],
                    Tuple[Sequence[_InputArgsType]],
                ]
            ]
        ] = None,
        skip_dynamic_shapes_check: bool = False,
    ):
        """Compare the results of PyTorch model with exported ONNX model

        Args:
            model (_ModelType): PyTorch model
            input_args (Sequence[_InputArgsType]): torch input arguments
            input_kwargs (Mapping[str, _InputArgsType]): torch input kwargs
            rtol (float, optional): relative tolerance. Defaults to 1e-3.
            atol (float, optional): absolute tolerance. Defaults to 1e-7.
            has_mutation (bool, optional): Whether the model mutates its input or state.
                `mutation` as `True` incurs extra overhead of cloning the inputs and model.
                Defaults to False.
            additional_test_inputs: Test the models with another dataset input, which
                is designed for dynamic axes testing. Defaults to None. It's a list of
                different input sets in tuples. Inside tuple, the first element is a tuple
                of args, and the second element is a dict of kwargs. Remember to put comma
                even if the following element is not provided.
                For example,
                additional_test_inputs = [((args1, args2), {"kwargs":1}), ((args1,),), ((), {"kwargs":1})]
            skip_dynamic_shapes_check: Whether to skip dynamic shape check. Defaults to False.
                Must be used when tests do not produce dynamic shapes even when dynamic shape feature is enabled.
                This is needed because Torch Dynamo uses the dynamic_shapes flag as a hint, only.

        """
        from torch._dynamo import config as _dynamo_config

        # avoid mutable data structure
        if input_kwargs is None:
            input_kwargs = {}

        if (
            has_mutation
            and self.model_type
            != pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM
        ):
            ref_model = _try_clone_model(model)
            ref_input_args, ref_input_kwargs = _try_clone_inputs(
                input_args, input_kwargs
            )
        else:
            ref_model = model
            ref_input_args = input_args
            ref_input_kwargs = input_kwargs

        assert isinstance(ref_model, torch.nn.Module) or callable(
            ref_model
        ), "Model must be a torch.nn.Module or callable"
        if (
            self.model_type
            == pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM
        ):
            with _dynamo_config.patch(do_not_emit_runtime_asserts=True):
                ref_model = torch.export.export(ref_model, args=ref_input_args)
            if (
                self.dynamic_shapes
            ):  # TODO: Support dynamic shapes for torch.export.ExportedProgram
                #       https://github.com/pytorch/pytorch/issues/113705
                pytest.xfail(
                    reason="torch.export.ExportedProgram does not support dynamic shapes"
                )

        # Feed args and kwargs into exporter.
        # Note that exporter should flatten kwargs into positional args the exported model;
        # since ONNX doesn't represent kwargs.
        with _dynamo_config.patch(do_not_emit_runtime_asserts=True):
            onnx_program = torch.onnx.dynamo_export(
                ref_model,
                *ref_input_args,
                **ref_input_kwargs,
                export_options=torch.onnx.ExportOptions(
                    dynamic_shapes=self.dynamic_shapes,
                    diagnostic_options=torch.onnx.DiagnosticOptions(
                        verbosity_level=logging.DEBUG
                    ),
                ),
            )

        if not skip_dynamic_shapes_check:
            assert_dynamic_shapes(onnx_program, self.dynamic_shapes)

        if isinstance(ref_model, torch.export.ExportedProgram):
            ref_model = ref_model.module()

        _compare_pytorch_onnx_with_ort(
            onnx_program,
            ref_model,
            input_args,
            input_kwargs,
            atol,
            rtol,
            has_mutation=has_mutation,
        )
        # This confirms the exported mode accepts different input shapes
        # when dynamic shape is enabled.
        if additional_test_inputs and self.dynamic_shapes:
            for another_input in additional_test_inputs:
                if len(another_input) > 2:
                    raise ValueError(
                        f"test_inputs should only have tuple args and dictionary kwargs. But receives: {len(another_input)}"
                    )
                additional_input_args = another_input[0]
                additional_input_kwargs = (
                    another_input[1]
                    if len(another_input) == 2 and another_input[1] is not None
                    else {}
                )
                _compare_pytorch_onnx_with_ort(
                    onnx_program,
                    ref_model,
                    additional_input_args,
                    additional_input_kwargs,
                    atol,
                    rtol,
                    has_mutation=has_mutation,
                )


def run_ort(
    onnx_model: Union[str, torch.onnx.ONNXProgram],
    pytorch_inputs: Sequence[_InputArgsType],
) -> _OutputsType:
    """Run ORT on the given ONNX model and inputs

    Used in test_fx_to_onnx_with_onnxruntime.py

    Args:
        onnx_model (Union[str, torch.onnx.ONNXProgram]): Converter ONNX model
        pytorch_inputs (Sequence[_InputArgsType]): The given torch inputs

    Raises:
        AssertionError: ONNX and PyTorch should have the same input sizes

    Returns:
        _OutputsType: ONNX model predictions
    """
    if isinstance(onnx_model, torch.onnx.ONNXProgram):
        buffer = io.BytesIO()
        onnx_model.save(buffer)
        ort_model = buffer.getvalue()
    else:
        ort_model = onnx_model

    # Suppress floods of warnings from ONNX Runtime
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 3  # Error
    session = onnxruntime.InferenceSession(
        ort_model, providers=["CPUExecutionProvider"], sess_options=session_options
    )
    input_names = [ort_input.name for ort_input in session.get_inputs()]

    if len(input_names) != len(pytorch_inputs):
        raise AssertionError(
            f"Expected {len(input_names)} inputs, got {len(pytorch_inputs)}"
        )

    ort_input = {
        k: torch.Tensor.numpy(v, force=True)
        for k, v in zip(input_names, pytorch_inputs)
    }
    return session.run(None, ort_input)


def _try_clone_model(model: _ModelType) -> _ModelType:
    """Used for preserving original model in case forward mutates model states."""
    try:
        return copy.deepcopy(model)
    except Exception:
        warnings.warn(
            "Failed to clone model. Model state might be mutated during verification."
        )
        return model


def _try_clone_inputs(input_args, input_kwargs):
    ref_input_args = copy.deepcopy(input_args)
    ref_input_kwargs = copy.deepcopy(input_kwargs)
    return ref_input_args, ref_input_kwargs


def _compare_pytorch_onnx_with_ort(
    onnx_program: torch.onnx.ONNXProgram,
    model: _ModelType,
    input_args: Sequence[_InputArgsType],
    input_kwargs: Mapping[str, _InputArgsType],
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    has_mutation: bool = False,
):
    if has_mutation:
        ref_model = _try_clone_model(model)
        ref_input_args, ref_input_kwargs = _try_clone_inputs(input_args, input_kwargs)
    else:
        ref_model = model
        ref_input_args = input_args
        ref_input_kwargs = input_kwargs

    # NOTE: ONNXProgram holds a reference (not copy) to the original ref_model, including its state_dict.
    # Thus, ONNXProgram() must run before ref_model() to prevent ref_model.forward() from changing the state_dict.
    # Otherwise, the ref_model can change buffers on state_dict which would be used by ONNXProgram.__call__()
    # NOTE: `model_with_state_dict=ref_model` is specified to cover runs with FakeTensor support
    onnx_outputs = onnx_program(*input_args, **input_kwargs)
    ref_outputs = ref_model(*ref_input_args, **ref_input_kwargs)
    torch.testing.assert_close(onnx_outputs, ref_outputs, rtol=rtol, atol=atol)


# The min onnx opset version to test for
MIN_ONNX_OPSET_VERSION = 9
# The max onnx opset version to test for
MAX_ONNX_OPSET_VERSION = _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
TESTED_OPSETS = range(MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION + 1)

# The min onnx opset version to test for
FX_MIN_ONNX_OPSET_VERSION = 18
# The max onnx opset version to test for
FX_MAX_ONNX_OPSET_VERSION = 18
FX_TESTED_OPSETS = range(FX_MIN_ONNX_OPSET_VERSION, FX_MAX_ONNX_OPSET_VERSION + 1)

BOOL_TYPES = (torch.bool,)

INT_TYPES = (
    # torch.int8,
    # torch.int16,
    torch.int32,
    torch.int64,
    # torch.uint8,
)

QINT_TYPES = (
    torch.qint8,
    torch.quint8,
)

FLOAT_TYPES = (
    torch.float16,
    torch.float32,
    # torch.float64,  ORT doesn't support
)

COMPLEX_TYPES = (
    # torch.complex32,  NOTE: torch.complex32 is experimental in torch
    torch.complex64,
    # torch.complex128,  ORT doesn't support
)

TESTED_DTYPES = (
    # Boolean
    torch.bool,
    # Integers
    *INT_TYPES,
    # Floating types
    *FLOAT_TYPES,
    # Complex types
    *COMPLEX_TYPES,
)


@dataclasses.dataclass
class DecorateMeta:
    """Information about a test case to skip or xfail.

    Adapted from functorch: functorch/test/common_utils.py

    Attributes:
        op_name: The name of the operator.
        variant_name: The name of the OpInfo variant.
        decorator: The decorator to apply to the test case.
        opsets: The opsets to apply the decorator to.
        dtypes: The dtypes to apply the decorator to.
        reason: The reason for skipping.
        test_behavior: The behavior of the test case. [skip or xfail]
        matcher: The matcher to apply to the test case.
        enabled_if: Whether to enable test behavior. Usually used on onnx/ort version control
        model_type: The type of the torch model. Defaults to None.
    """

    op_name: str
    variant_name: str
    decorator: Callable
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]]
    dtypes: Optional[Collection[torch.dtype]]
    reason: str
    test_behavior: str
    matcher: Optional[Callable[[Any], bool]] = None
    enabled_if: bool = True
    model_type: Optional[pytorch_test_common.TorchModelType] = None

    def contains_opset(self, opset: int) -> bool:
        if self.opsets is None:
            return True
        return any(
            opset == opset_spec if isinstance(opset_spec, int) else opset_spec(opset)
            for opset_spec in self.opsets
        )


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], bool]] = None,
    enabled_if: bool = True,
    model_type: Optional[pytorch_test_common.TorchModelType] = None,
):
    """Expects a OpInfo test to fail.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            xfail is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether to enable xfail. Usually used on onnx/ort version control
        model_type: The type of the torch model. Defaults to None.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        opsets=opsets,
        dtypes=dtypes,
        enabled_if=enabled_if,
        matcher=matcher,
        reason=reason,
        test_behavior="xfail",
        model_type=model_type,
    )


def skip(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], Any]] = None,
    enabled_if: bool = True,
    model_type: Optional[pytorch_test_common.TorchModelType] = None,
):
    """Skips a test case in OpInfo that we don't care about.

    Likely because ONNX does not support the use case or it is by design.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            skip is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether to enable skip. Usually used on onnx/ort version control
        model_type: The type of the torch model. Defaults to None.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Skip: {reason}"),
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
        enabled_if=enabled_if,
        test_behavior="skip",
        model_type=model_type,
    )


def skip_slow(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], Any]] = None,
    model_type: Optional[pytorch_test_common.TorchModelType] = None,
):
    """Skips a test case in OpInfo that is too slow.

    It needs further investigation to understand why it is slow.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            skip is in the SKIP_XFAIL_SUBTESTS list.
        model_type: The type of the torch model. Defaults to None.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=common_utils.slowTest,
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
        enabled_if=not common_utils.TEST_WITH_SLOW,
        test_behavior="skip",
        model_type=model_type,
    )


def add_decorate_info(
    all_opinfos: Sequence[opinfo_core.OpInfo],
    test_class_name: str,
    base_test_name: str,
    opset: int,
    skip_or_xfails: Iterable[DecorateMeta],
):
    """Decorates OpInfo tests with decorators based on the skip_or_xfails list.

    Args:
        all_opinfos: All OpInfos.
        test_class_name: The name of the test class.
        base_test_name: The name of the test method.
        opset: The opset to decorate for.
        skip_or_xfails: DecorateMeta's.
    """
    ops_mapping = {(info.name, info.variant_test_name): info for info in all_opinfos}
    for decorate_meta in skip_or_xfails:
        if not decorate_meta.contains_opset(opset):
            # Skip does not apply to this opset
            continue
        opinfo = ops_mapping.get((decorate_meta.op_name, decorate_meta.variant_name))
        assert (
            opinfo is not None
        ), f"Couldn't find OpInfo for {decorate_meta}. Did you need to specify variant_name?"
        assert decorate_meta.model_type is None, (
            f"Tested op: {decorate_meta.op_name} in wrong position! "
            "If model_type needs to be specified, it should be "
            "put under SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE."
        )
        decorators = list(opinfo.decorators)
        new_decorator = opinfo_core.DecorateInfo(
            decorate_meta.decorator,
            test_class_name,
            base_test_name,
            dtypes=decorate_meta.dtypes,
            active_if=decorate_meta.enabled_if,
        )
        decorators.append(new_decorator)
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn

    return wrapped


def opsets_before(opset: int) -> Callable[[int], bool]:
    """Returns a comparison function that decides if the given opset is before the specified."""

    def compare(other_opset: int):
        return other_opset < opset

    return compare


def opsets_after(opset: int) -> Callable[[int], bool]:
    """Returns a comparison function that decides if the given opset is after the specified."""

    def compare(other_opset: int):
        return other_opset > opset

    return compare


def reason_onnx_script_does_not_support(
    operator: str, dtypes: Optional[Sequence[str]] = None
) -> str:
    """Formats the reason: ONNX script doesn't support the given dtypes."""
    return f"{operator} on {dtypes or 'dtypes'} not supported by ONNX script"


def reason_onnx_runtime_does_not_support(
    operator: str, dtypes: Optional[Sequence[str]] = None
) -> str:
    """Formats the reason: ONNX Runtime doesn't support the given dtypes."""
    return f"{operator} on {dtypes or 'dtypes'} not supported by ONNX Runtime"


def reason_onnx_does_not_support(
    operator: str, dtypes: Optional[Sequence[str]] = None
) -> str:
    """Formats the reason: ONNX doesn't support the given dtypes."""
    return f"{operator} on {dtypes or 'certain dtypes'} not supported by the ONNX Spec"


def reason_dynamo_does_not_support(
    operator: str, dtypes: Optional[Sequence[str]] = None
) -> str:
    """Formats the reason: Dynamo doesn't support the given dtypes."""
    return (
        f"{operator} on {dtypes or 'certain dtypes'} not supported by the Dynamo Spec"
    )


def reason_jit_tracer_error(info: str) -> str:
    """Formats the reason: JIT tracer errors."""
    return f"JIT tracer error on {info}"


def reason_flaky() -> str:
    """Formats the reason: test is flaky."""
    return "flaky test"


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
    except Exception as e:  # pylint: disable=broad-exception-caught
        if test_behavior is None:
            raise e
        if test_behavior == "xfail":
            pytest.xfail(reason=reason)
    else:
        if test_behavior == "xfail":
            pytest.fail("Test unexpectedly passed")
