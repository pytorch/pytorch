## @package test_util
# Module caffe2.python.test_util
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import core, workspace

import unittest
import os


def rand_array(*dims):
    # np.random.rand() returns float instead of 0-dim array, that's why need to
    # do some tricks
    return np.array(np.random.rand(*dims) - 0.5).astype(np.float32)


def randBlob(name, type, *dims, **kwargs):
    offset = kwargs['offset'] if 'offset' in kwargs else 0.0
    workspace.FeedBlob(name, np.random.rand(*dims).astype(type) + offset)


def randBlobFloat32(name, *dims, **kwargs):
    randBlob(name, np.float32, *dims, **kwargs)


def randBlobsFloat32(names, *dims, **kwargs):
    for name in names:
        randBlobFloat32(name, *dims, **kwargs)


def numOps(net):
    return len(net.Proto().op)


def str_compare(a, b, encoding="utf8"):
    if isinstance(a, bytes):
        a = a.decode(encoding)
    if isinstance(b, bytes):
        b = b.decode(encoding)
    return a == b


def get_default_test_flags():
    return [
        'caffe2',
        '--caffe2_log_level=0',
        '--caffe2_cpu_allocator_do_zero_fill=0',
        '--caffe2_cpu_allocator_do_junk_fill=1',
    ]


def caffe2_flaky(test_method):
    # This decorator is used to mark a test method as flaky.
    # This is used in conjunction with the environment variable
    # CAFFE2_RUN_FLAKY_TESTS that specifies "flaky tests" mode
    # If flaky tests mode are on, only flaky tests are run
    # If flaky tests mode are off, only non-flaky tests are run
    # NOTE: the decorator should be applied as the top-level decorator
    # in a test method.
    test_method.__caffe2_flaky__ = True
    return test_method


def is_flaky_test_mode():
    return os.getenv('CAFFE2_RUN_FLAKY_TESTS', '0') == '1'


class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workspace.GlobalInit(get_default_test_flags())
        # clear the default engines settings to separate out its
        # affect from the ops tests
        core.SetEnginePref({}, {})

    def setUp(self):
        # Skip tests based on whether we're in flaky test mode and
        # the test is decorated as a flaky test.
        test_method = getattr(self, self._testMethodName)
        is_flaky_test = getattr(test_method, "__caffe2_flaky__", False)
        if (is_flaky_test_mode() and not is_flaky_test):
            raise unittest.SkipTest("Non-flaky tests are skipped in flaky test mode")
        elif (not is_flaky_test_mode() and is_flaky_test):
            raise unittest.SkipTest("Flaky tests are skipped in regular test mode")

        self.ws = workspace.C.Workspace()
        workspace.ResetWorkspace()

    def tearDown(self):
        workspace.ResetWorkspace()
