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


class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workspace.GlobalInit([
            'caffe2',
            '--caffe2_log_level=0',
            '--caffe2_cpu_allocator_do_zero_fill=0',
            '--caffe2_cpu_allocator_do_junk_fill=1',
        ])
        # clear the default engines settings to separate out its
        # affect from the ops tests
        core.SetEnginePref({}, {})

    def setUp(self):
        self.ws = workspace.C.Workspace()
        workspace.ResetWorkspace()

    def tearDown(self):
        workspace.ResetWorkspace()
