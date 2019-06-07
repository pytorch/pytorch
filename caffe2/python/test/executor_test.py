from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import core, workspace
from caffe2.python.test.executor_test_util import (
    build_conv_model,
    build_resnet50_dataparallel_model,
    run_resnet50_epoch,
    ExecutorTestBase,
    executor_test_settings,
    executor_test_model_names)

from caffe2.python.test_util import TestCase

from hypothesis import given
import hypothesis.strategies as st

import unittest


EXECUTORS = ["parallel", "async_scheduling"]
ITERATIONS = 1


class ExecutorCPUConvNetTest(ExecutorTestBase):
    @given(executor=st.sampled_from(EXECUTORS),
           model_name=st.sampled_from(executor_test_model_names()),
           batch_size=st.sampled_from([1]),
           num_workers=st.sampled_from([8]))
    @executor_test_settings
    def test_executor(self, executor, model_name, batch_size, num_workers):
        model = build_conv_model(model_name, batch_size)
        model.Proto().num_workers = num_workers

        def run_model():
            iterations = ITERATIONS
            if model_name == "MLP":
                iterations = 1  # avoid numeric instability with MLP gradients
            workspace.RunNet(model.net, iterations)

        self.compare_executors(
            model,
            ref_executor="simple",
            test_executor=executor,
            model_run_func=run_model,
        )


@unittest.skipIf(not workspace.has_gpu_support, "no gpu")
class ExecutorGPUResNetTest(ExecutorTestBase):
    @given(executor=st.sampled_from(EXECUTORS),
           num_workers=st.sampled_from([8]))
    @executor_test_settings
    def test_executor(self, executor, num_workers):
        model = build_resnet50_dataparallel_model(
            num_gpus=workspace.NumGpuDevices(), batch_size=8, epoch_size=8)
        model.Proto().num_workers = num_workers

        def run_model():
            run_resnet50_epoch(model, batch_size=8, epoch_size=8)

        self.compare_executors(
            model,
            ref_executor="simple",
            test_executor=executor,
            model_run_func=run_model,
        )


class ExecutorFailingOpTest(TestCase):
    def test_failing_op(self):
        def create_failing_net(throw_exception):
            net = core.Net("failing_net")
            if throw_exception:
                net.ThrowException([], [])
            else:
                net.Fail([], [])
            net.Proto().type = "async_scheduling"
            return net

        workspace.ResetWorkspace()
        net = create_failing_net(throw_exception=True)
        workspace.CreateNet(net)
        with self.assertRaises(RuntimeError):
            workspace.RunNet(net)

        with self.assertRaises(RuntimeError):
            workspace.RunNet(net, allow_fail=True)

        workspace.ResetWorkspace()
        net = create_failing_net(throw_exception=False)
        workspace.CreateNet(net)

        with self.assertRaises(RuntimeError):
            workspace.RunNet(net)

        res = workspace.RunNet(net, allow_fail=True)
        self.assertFalse(res)


if __name__ == '__main__':
    unittest.main()
