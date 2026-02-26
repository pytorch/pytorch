# Owner(s): ["module: onnx"]

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import unittest
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np
import onnxruntime
import pytest
import pytorch_test_common

import torch
from torch import export as torch_export
from torch.onnx import _constants
from torch.onnx._internal.torchscript_exporter import verification
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
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
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


def parameterize_class_name(cls: type, idx: int, input_dicts: Mapping[Any, Any]):
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


# The min onnx opset version to test for
MIN_ONNX_OPSET_VERSION = 9
# The max onnx opset version to test for
MAX_ONNX_OPSET_VERSION = _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
TESTED_OPSETS = range(MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION + 1)

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
        if opinfo is None:
            raise AssertionError(
                f"Couldn't find OpInfo for {decorate_meta}. Did you need to specify variant_name?"
            )
        if decorate_meta.model_type is not None:
            raise AssertionError(
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
