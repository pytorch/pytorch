from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import workspace
from caffe2.python.test.executor_test_util import (
    conv_model_generators,
    build_conv_model,
    build_resnet50_dataparallel_model,
    run_resnet50_epoch,
    ExecutorTestBase)

from hypothesis import given, settings
import hypothesis.strategies as st

from caffe2.python import hypothesis_test_util as hu

import unittest


EXECUTORS = ["dag", "async_dag"]
ITERATIONS = 2
SANDCASTLE_MAX_EXAMPLES = 2
SANDCASTLE_TIMEOUT = 600


def sandcastle_settings(func):
    if hu.is_sandcastle():
        return settings(
            max_examples=SANDCASTLE_MAX_EXAMPLES,
            timeout=SANDCASTLE_TIMEOUT
        )(func)
    else:
        return func


class ExecutorCPUConvNetTest(ExecutorTestBase):
    @given(executor=st.sampled_from(EXECUTORS),
           model_name=st.sampled_from(conv_model_generators().keys()),
           batch_size=st.sampled_from([8]),
           num_workers=st.sampled_from([8]))
    @sandcastle_settings
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
    @sandcastle_settings
    def test_executor(self, executor, num_workers):
        model = build_resnet50_dataparallel_model(
            num_gpus=workspace.NumCudaDevices(), batch_size=32, epoch_size=32)
        model.Proto().num_workers = num_workers

        def run_model():
            run_resnet50_epoch(model, batch_size=32, epoch_size=32)

        self.compare_executors(
            model,
            ref_executor="simple",
            test_executor=executor,
            model_run_func=run_model,
        )


if __name__ == '__main__':
    unittest.main()
