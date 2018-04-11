## @package test_util
# Module caffe2.python.test_util
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import core, workspace

import unittest


def rand_array(*dims):
    # np.random.rand() returns float instead of 0-dim array, that's why need to
    # do some tricks
    return np.array(np.random.rand(*dims) - 0.5).astype(np.float32)


class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workspace.GlobalInit([
            'caffe2',
            '--caffe2_log_level=0',
        ])
        # clear the default engines settings to separate out its
        # affect from the ops tests
        core.SetEnginePref({}, {})

    def setUp(self):
        self.ws = workspace.C.Workspace()
        workspace.ResetWorkspace()

    def tearDown(self):
        workspace.ResetWorkspace()
