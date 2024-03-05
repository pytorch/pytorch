# Owner(s): ["module: onnx"]
from __future__ import annotations

import functools
import os
import random
import sys
import unittest
from enum import auto, Enum
from typing import Optional

import numpy as np
import packaging.version
import pytest

import torch
from torch.autograd import function
from torch.onnx._internal import diagnostics
from torch.testing._internal import common_utils

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, pytorch_test_dir)

torch.set_default_dtype(torch.float)

BATCH_SIZE = 2

RNN_BATCH_SIZE = 7
RNN_SEQUENCE_LENGTH = 11
RNN_INPUT_SIZE = 5
RNN_HIDDEN_SIZE = 3


class TorchModelType(Enum):
    TORCH_NN_MODULE = auto()
    TORCH_EXPORT_EXPORTEDPROGRAM = auto()


def _skipper(condition, reason):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if condition():
                raise unittest.SkipTest(reason)
            return f(*args, **kwargs)

        return wrapper

    return decorator


skipIfNoCuda = _skipper(lambda: not torch.cuda.is_available(), "CUDA is not available")

skipIfTravis = _skipper(lambda: os.getenv("TRAVIS"), "Skip In Travis")

skipIfNoBFloat16Cuda = _skipper(
    lambda: not torch.cuda.is_bf16_supported(), "BFloat16 CUDA is not available"
)

skipIfQuantizationBackendQNNPack = _skipper(
    lambda: torch.backends.quantized.engine == "qnnpack",
    "Not compatible with QNNPack quantization backend",
)


# skips tests for all versions below min_opset_version.
# if exporting the op is only supported after a specific version,
# add this wrapper to prevent running the test for opset_versions
# smaller than the currently tested opset_version
def skipIfUnsupportedMinOpsetVersion(min_opset_version):
    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.opset_version < min_opset_version:
                raise unittest.SkipTest(
                    f"Unsupported opset_version: {self.opset_version} < {min_opset_version}"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


# skips tests for all versions above max_opset_version.
def skipIfUnsupportedMaxOpsetVersion(max_opset_version):
    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.opset_version > max_opset_version:
                raise unittest.SkipTest(
                    f"Unsupported opset_version: {self.opset_version} > {max_opset_version}"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


# skips tests for all opset versions.
def skipForAllOpsetVersions():
    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.opset_version:
                raise unittest.SkipTest(
                    "Skip verify test for unsupported opset_version"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def skipTraceTest(skip_before_opset_version: Optional[int] = None, reason: str = ""):
    """Skip tracing test for opset version less than skip_before_opset_version.

    Args:
        skip_before_opset_version: The opset version before which to skip tracing test.
            If None, tracing test is always skipped.
        reason: The reason for skipping tracing test.

    Returns:
        A decorator for skipping tracing test.
    """

    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if skip_before_opset_version is not None:
                self.skip_this_opset = self.opset_version < skip_before_opset_version
            else:
                self.skip_this_opset = True
            if self.skip_this_opset and not self.is_script:
                raise unittest.SkipTest(f"Skip verify test for torch trace. {reason}")
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def skipScriptTest(skip_before_opset_version: Optional[int] = None, reason: str = ""):
    """Skip scripting test for opset version less than skip_before_opset_version.

    Args:
        skip_before_opset_version: The opset version before which to skip scripting test.
            If None, scripting test is always skipped.
        reason: The reason for skipping scripting test.

    Returns:
        A decorator for skipping scripting test.
    """

    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if skip_before_opset_version is not None:
                self.skip_this_opset = self.opset_version < skip_before_opset_version
            else:
                self.skip_this_opset = True
            if self.skip_this_opset and self.is_script:
                raise unittest.SkipTest(f"Skip verify test for TorchScript. {reason}")
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


# NOTE: This decorator is currently unused, but we may want to use it in the future when
# we have more tests that are not supported in released ORT.
def skip_min_ort_version(reason: str, version: str, dynamic_only: bool = False):
    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if (
                packaging.version.parse(self.ort_version).release
                < packaging.version.parse(version).release
            ):
                if dynamic_only and not self.dynamic_shapes:
                    return func(self, *args, **kwargs)

                raise unittest.SkipTest(
                    f"ONNX Runtime version: {version} is older than required version {version}. "
                    f"Reason: {reason}."
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def xfail_dynamic_fx_test(
    error_message: str,
    model_type: Optional[TorchModelType] = None,
    reason: Optional[str] = None,
):
    """Xfail dynamic exporting test.

    Args:
        reason: The reason for xfailing dynamic exporting test.
        model_type (TorchModelType): The model type to xfail dynamic exporting test for.
            When None, model type is not used to skip dynamic tests.

    Returns:
        A decorator for skipping dynamic exporting test.
    """

    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.dynamic_shapes and (
                not model_type or self.model_type == model_type
            ):
                return xfail(error_message, reason)(func)(self, *args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def skip_dynamic_fx_test(reason: str, model_type: TorchModelType = None):
    """Skip dynamic exporting test.

    Args:
        reason: The reason for skipping dynamic exporting test.
        model_type (TorchModelType): The model type to skip dynamic exporting test for.
            When None, model type is not used to skip dynamic tests.

    Returns:
        A decorator for skipping dynamic exporting test.
    """

    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.dynamic_shapes and (
                not model_type or self.model_type == model_type
            ):
                raise unittest.SkipTest(
                    f"Skip verify dynamic shapes test for FX. {reason}"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def skip_in_ci(reason: str):
    """Skip test in CI.

    Args:
        reason: The reason for skipping test in CI.

    Returns:
        A decorator for skipping test in CI.
    """

    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if os.getenv("CI"):
                raise unittest.SkipTest(f"Skip test in CI. {reason}")
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def xfail(error_message: str, reason: Optional[str] = None):
    """Expect failure.

    Args:
        reason: The reason for expected failure.

    Returns:
        A decorator for expecting test failure.
    """

    def wrapper(func):
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            try:
                func(self, *args, **kwargs)
            except Exception as e:
                if isinstance(e, torch.onnx.OnnxExporterError):
                    # diagnostic message is in the cause of the exception
                    assert error_message in str(
                        e.__cause__
                    ), f"Expected error message: {error_message} NOT in {str(e.__cause__)}"
                else:
                    assert error_message in str(
                        e
                    ), f"Expected error message: {error_message} NOT in {str(e)}"
                pytest.xfail(reason if reason else f"Expected failure: {error_message}")
            else:
                pytest.fail("Unexpected success!")

        return inner

    return wrapper


# skips tests for opset_versions listed in unsupported_opset_versions.
# if the caffe2 test cannot be run for a specific version, add this wrapper
# (for example, an op was modified but the change is not supported in caffe2)
def skipIfUnsupportedOpsetVersion(unsupported_opset_versions):
    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.opset_version in unsupported_opset_versions:
                raise unittest.SkipTest(
                    "Skip verify test for unsupported opset_version"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def skipShapeChecking(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.check_shape = False
        return func(self, *args, **kwargs)

    return wrapper


def skipDtypeChecking(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.check_dtype = False
        return func(self, *args, **kwargs)

    return wrapper


def xfail_if_model_type_is_exportedprogram(
    error_message: str, reason: Optional[str] = None
):
    """xfail test with models using ExportedProgram as input.

    Args:
        error_message: The error message to raise when the test is xfailed.
        reason: The reason for xfail the ONNX export test.

    Returns:
        A decorator for xfail tests.
    """

    def xfail_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.model_type == TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM:
                return xfail(error_message, reason)(func)(self, *args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return xfail_dec


def xfail_if_model_type_is_not_exportedprogram(
    error_message: str, reason: Optional[str] = None
):
    """xfail test without models using ExportedProgram as input.

    Args:
        reason: The reason for xfail the ONNX export test.

    Returns:
        A decorator for xfail tests.
    """

    def xfail_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.model_type != TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM:
                return xfail(error_message, reason)(func)(self, *args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return xfail_dec


def flatten(x):
    return tuple(function._iter_filter(lambda o: isinstance(o, torch.Tensor))(x))


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class ExportTestCase(common_utils.TestCase):
    """Test case for ONNX export.

    Any test case that tests functionalities under torch.onnx should inherit from this class.
    """

    def setUp(self):
        super().setUp()
        # TODO(#88264): Flaky test failures after changing seed.
        set_rng_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        diagnostics.engine.clear()
