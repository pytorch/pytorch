from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import os
import unittest
import sys
import torch
import torch.autograd.function as function

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, pytorch_test_dir)

from torch.testing._internal.common_utils import *  # noqa: F401

torch.set_default_tensor_type('torch.FloatTensor')

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


skipIfNoCuda = _skipper(lambda: not torch.cuda.is_available(),
                        'CUDA is not available')

skipIfTravis = _skipper(lambda: os.getenv('TRAVIS'),
                        'Skip In Travis')

# skips tests for all versions below min_opset_version.
# if exporting the op is only supported after a specific version,
# add this wrapper to prevent running the test for opset_versions
# smaller than the currently tested opset_version
def skipIfUnsupportedMinOpsetVersion(min_opset_version):
    def skip_dec(func):
        def wrapper(self):
            if self.opset_version < min_opset_version:
                raise unittest.SkipTest("Skip verify test for unsupported opset_version")
            return func(self)
        return wrapper
    return skip_dec

# Enables tests for scripting, instead of only tracing the model.
def enableScriptTest():
    def script_dec(func):
        def wrapper(self):
            self.is_script_test_enabled = True
            return func(self)
        return wrapper
    return script_dec

# skips tests for opset_versions listed in unsupported_opset_versions.
# if the caffe2 test cannot be run for a specific version, add this wrapper
# (for example, an op was modified but the change is not supported in caffe2)
def skipIfUnsupportedOpsetVersion(unsupported_opset_versions):
    def skip_dec(func):
        def wrapper(self):
            if self.opset_version in unsupported_opset_versions:
                raise unittest.SkipTest("Skip verify test for unsupported opset_version")
            return func(self)
        return wrapper
    return skip_dec

def flatten(x):
    return tuple(function._iter_filter(lambda o: isinstance(o, torch.Tensor))(x))
