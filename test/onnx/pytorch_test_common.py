# Owner(s): ["module: onnx"]

import functools
import os
import sys
import unittest
from typing import Optional

import torch
from torch.autograd import function

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
