# Owner(s): ["module: onnx"]
from __future__ import annotations

import functools
import os
import random
import sys
import unittest
from typing import Optional

import numpy as np
import packaging.version

import torch
from torch.autograd import function
from torch.onnx._internal import diagnostics
from torch.testing._internal import common_utils

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, pytorch_test_dir)

torch.set_default_tensor_type("torch.FloatTensor")

BATCH_SIZE = 2

RNN_BATCH_SIZE = 7
RNN_SEQUENCE_LENGTH = 11
RNN_INPUT_SIZE = 5
RNN_HIDDEN_SIZE = 3


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


# TODO(titaiwang): dynamic_only is specific to the situation that dynamic fx exporter
# is not yet supported by ORT until 1.15.0. Remove dynamic_only once ORT 1.15.0 is released.
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


def skip_dynamic_fx_test(reason: str):
    """Skip dynamic exporting test.

    Args:
        reason: The reason for skipping dynamic exporting test.

    Returns:
        A decorator for skipping dynamic exporting test.
    """

    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.dynamic_shapes:
                raise unittest.SkipTest(
                    f"Skip verify dynamic shapes test for FX. {reason}"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return skip_dec


def skip_op_level_debug_test(reason: str):
    """Skip tests with op_level_debug enabled.

    Args:
        reason: The reason for skipping tests with op_level_debug enabled.

    Returns:
        A decorator for skipping tests with op_level_debug enabled.
    """

    def skip_dec(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.op_level_debug:
                raise unittest.SkipTest(
                    f"Skip test with op_level_debug enabled. {reason}"
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


def xfail(reason: str):
    """Expect failure.

    Args:
        reason: The reason for expected failure.

    Returns:
        A decorator for expecting test failure.
    """
    return unittest.expectedFailure


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
