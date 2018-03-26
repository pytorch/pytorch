from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, dyndep
import caffe2.python.hypothesis_test_util as hu

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import os
import unittest

try:
    from libfb.py import parutil
except ImportError as e:
    # If libfb not found, skip all tests in this file
    raise unittest.SkipTest(str(e))

core.GlobalInit(["python", "--caffe2_log_level=0"])

dyndep.InitOpsLibrary('@/caffe2/caffe2/contrib/torch:torch_ops')

RUNTIME = parutil.get_runtime_path()
if 'LUA_PATH' not in os.environ:
    os.environ['LUA_PATH'] = ";".join([
        os.path.join(RUNTIME, '_lua', '?.lua'),
        os.path.join(RUNTIME, '_lua', '?', 'init.lua'),
    ])
    os.environ['LUA_CPATH'] = os.path.join(RUNTIME, '_lua', '?.so')


class TorchOpTest(hu.HypothesisTestCase):
    @given(n=st.integers(min_value=1, max_value=10),
           i=st.integers(min_value=1, max_value=10),
           h=st.integers(min_value=2, max_value=10))
    def test_feed(self, n, i, h):
        op = core.CreateOperator(
            "Torch", ["x", "W", "b"], ["y"],
            init=b"nn.Linear({i}, {h})".format(h=h, i=i),
            num_inputs=1,
            num_params=2,
            num_outputs=1
        )
        x = np.random.randn(n, i).astype(np.float32)
        W = np.random.randn(h, i).astype(np.float32)
        b = np.random.randn(h).astype(np.float32)
        self.ws.create_blob("x").feed(x)
        self.ws.create_blob("W").feed(W)
        self.ws.create_blob("b").feed(b)
        self.ws.run(op)
        y = self.ws.blobs["y"].fetch()
        print("y", y)
        y = y.reshape((n, h))
        np.testing.assert_allclose(y, np.dot(x, W.T) + b, atol=1e-4, rtol=1e-4)

    @given(n=st.integers(min_value=1, max_value=10),
           i=st.integers(min_value=1, max_value=10),
           h=st.integers(min_value=2, max_value=10),
           **hu.gcs)
    def test_gradient(self, n, i, h, gc, dc):
        op = core.CreateOperator(
            "Torch", ["x", "W", "b"], ["y"],
            init=b"nn.Linear({i}, {h})".format(h=h, i=i),
            num_inputs=1,
            num_params=2,
            num_outputs=1
        )
        x = np.random.randn(n, i).astype(np.float32)
        W = np.random.randn(h, i).astype(np.float32)
        b = np.random.randn(h).astype(np.float32)
        inputs = [x, W, b]
        self.assertDeviceChecks(dc, op, inputs, [0])
        for i, _ in enumerate(inputs):
            self.assertGradientChecks(gc, op, inputs, i, [0])

    @given(n=st.integers(min_value=1, max_value=10),
           i=st.integers(min_value=1, max_value=10),
           h=st.integers(min_value=2, max_value=10),
           iters=st.integers(min_value=1, max_value=100))
    def test_iterated(self, n, i, h, iters):
        x = np.random.randn(n, i).astype(np.float32)
        W = np.random.randn(h, i).astype(np.float32)
        b = np.random.randn(h).astype(np.float32)
        self.ws.create_blob("x").feed(x)
        self.ws.create_blob("W").feed(W)
        self.ws.create_blob("b").feed(b)
        net = core.Net("op")
        net.Torch(
            ["x", "W", "b"], ["y"],
            init=b"nn.Linear({i}, {h})".format(h=h, i=i),
            num_inputs=1,
            num_params=2,
            num_outputs=1
        )
        print(net.Proto())
        net_ = self.ws.create_net(net)
        for i in range(iters):
            if i % 1000 == 0:
                print(i)
            net_.run()

        y = self.ws.blobs["y"].fetch()
        y = y.reshape((n, h))
        np.testing.assert_allclose(y, np.dot(x, W.T) + b, atol=1e-4, rtol=1e-4)

    def test_leakage_torch(self):
        n = 1
        i = 100
        h = 1000
        iters = 2000
        x = np.random.randn(n, i).astype(np.float32)
        W = np.random.randn(h, i).astype(np.float32)
        b = np.random.randn(h).astype(np.float32)
        self.ws.create_blob("x").feed(x)
        self.ws.create_blob("W").feed(W)
        self.ws.create_blob("b").feed(b)
        net = core.Net("op")
        net.Torch(
            ["x", "W", "b"], ["y"],
            init=b"nn.Linear({i}, {h})".format(h=h, i=i),
            num_inputs=1,
            num_params=2,
            num_outputs=1
        )
        net_ = self.ws.create_net(net)
        for i in range(iters):
            if i % 1000 == 0:
                print(i)
            net_.run()

        y = self.ws.blobs["y"].fetch()
        y = y.reshape((n, h))
        np.testing.assert_allclose(y, np.dot(x, W.T) + b, atol=1e-4, rtol=1e-4)
