from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given, assume, settings
import numpy as np
import time
import os
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu

np.random.seed(1)


def benchmark(net, warmups=5, iters=100):
    for _ in range(warmups):
        workspace.RunNetOnce(net.Proto().SerializeToString())
    plan = core.Plan("plan")
    plan.AddStep(core.ExecutionStep("test-step", net, iters))
    before = time.time()
    workspace.RunPlan(plan.Proto().SerializeToString())
    after = time.time()
    print("Timing network, time taken per-iteration: {:.6f}ms".format((
        after - before) / float(iters) * 1000.0))
    return after - before


def has_avx2():
    import subprocess
    try:
        subprocess.check_output(["grep", "avx2", "/proc/cpuinfo"])
        return True
    except subprocess.CalledProcessError:
        # grep exits with rc 1 on no matches
        return False


@unittest.skipIf(not has_avx2(), "NNPACK requires AVX2")
class NNPackOpsTest(hu.HypothesisTestCase):
    @given(stride=st.integers(1, 1),
           pad=st.integers(0, 2),
           kernel=st.integers(3, 5),
           size=st.integers(5, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 5))
    def test_convolution_correctness(self, stride, pad, kernel, size,
                                     input_channels, output_channels,
                                     batch_size):
        assume(stride <= kernel)
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
            output_channels, input_channels, kernel, kernel).astype(np.float32)\
            - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        order = "NCHW"
        outputs = {}
        for engine in ["", "NNPACK"]:
            op = core.CreateOperator(
                "Conv",
                ["X", "w", "b"],
                ["Y"],
                stride=stride,
                kernel=kernel,
                pad=pad,
                order=order,
                kts="TUPLE",
                engine=engine,
            )
            workspace.FeedBlob("X", X)
            workspace.FeedBlob("w", w)
            workspace.FeedBlob("b", b)
            workspace.RunOperatorOnce(op)
            outputs[engine] = workspace.FetchBlob("Y")
        np.testing.assert_allclose(
            outputs[""],
            outputs["NNPACK"],
            atol=1e-4,
            rtol=1e-4)

    @given(size=st.sampled_from([6, 8]),
           input_channels=st.integers(1, 8),
           batch_size=st.integers(1, 5))
    def test_max_pool_correctness(self, size, input_channels, batch_size):
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        order = "NCHW"
        outputs = {}
        # only 2 * 2 stride and 2 * 2 pool is supported in NNPack now
        stride = 2
        kernel = 2
        # The pooling strategy of NNPack is different from caffe2 pooling
        pad = 0
        for engine in ["", "NNPACK"]:
            op = core.CreateOperator(
                "MaxPool",
                ["X"],
                ["Y"],
                stride=stride,
                kernel=kernel,
                pad=pad,
                order=order,
                engine=engine,
            )
            workspace.FeedBlob("X", X)
            workspace.RunOperatorOnce(op)
            outputs[engine] = workspace.FetchBlob("Y")
        np.testing.assert_allclose(
            outputs[""],
            outputs["NNPACK"],
            atol=1e-4,
            rtol=1e-4)

    @settings(timeout=3600)
    @unittest.skipIf(not os.environ.get("CAFFE2_BENCHMARK"), "Benchmark")
    @given(stride=st.integers(1, 1),
           pad=st.integers(0, 2),
           kernel=st.sampled_from([3, 5, 7]),
           size=st.integers(30, 90),
           input_channels=st.sampled_from([3, 64, 256]),
           output_channels=st.sampled_from([32, 96, 256]),
           batch_size=st.sampled_from([32, 64, 96, 128]))
    def test_timings(self, stride, pad, kernel, size,
                     input_channels, output_channels, batch_size):
        assume(stride <= kernel)
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        w = np.random.rand(output_channels, input_channels,
                           kernel, kernel).astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        order = "NCHW"
        times = {}
        for engine in ["", "NNPACK"]:
            net = core.Net(engine + "_test")
            net.Conv(
                ["X", "W", "b"], "Y",
                order=order,
                kernel=kernel,
                stride=stride,
                pad=pad,
                kts="TUPLE",
                engine=engine,
            )
            workspace.FeedBlob("X", X)
            workspace.FeedBlob("W", w)
            workspace.FeedBlob("b", b)
            workspace.RunNetOnce(net.Proto().SerializeToString())
            times[engine] = benchmark(net)
        print("Speedup for NNPACK: {:.2f}".format(
            times[""] / times["NNPACK"]))
