from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import workspace
from caffe2.python.test.executor_test_util import (
    build_conv_model,
    build_resnet50_dataparallel_model,
    run_resnet50_epoch,
    ExecutorTestBase,
    executor_test_settings,
    executor_test_model_names)

from hypothesis import given
import hypothesis.strategies as st

import unittest


EXECUTORS = ["async_scheduling", "async_polling", "dag", "async_dag"]
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
            num_gpus=workspace.NumCudaDevices(), batch_size=8, epoch_size=8)
        model.Proto().num_workers = num_workers

        def run_model():
            run_resnet50_epoch(model, batch_size=8, epoch_size=8)

        self.compare_executors(
            model,
            ref_executor="simple",
            test_executor=executor,
            model_run_func=run_model,
        )


if __name__ == '__main__':
    unittest.main()
