from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import contextlib
import hypothesis
import hypothesis.strategies as st
import hypothesis.extra.numpy
import unittest
import os
import numpy as np
from caffe2.python import workspace, device_checker, gradient_checker, test_util
from caffe2.proto import caffe2_pb2


def is_sandcastle():
    if os.getenv('SANDCASTLE') == '1':
        return True
    elif os.getenv('TW_JOB_USER') == 'sandcastle':
        return True
    return False

hypothesis.settings.register_profile(
    "sandcastle", hypothesis.settings(max_examples=100))
hypothesis.settings.register_profile(
    "dev", hypothesis.settings(max_examples=2))
hypothesis.settings.register_profile(
    "debug", hypothesis.settings(
        max_examples=1000, verbosity=hypothesis.Verbosity.verbose))
hypothesis.settings.load_profile(
    'sandcastle' if is_sandcastle()
    else os.getenv('CAFFE2_HYPOTHESIS_PROFILE', 'dev'))


def dims(min_value=1, max_value=5):
    return st.integers(min_value=min_value, max_value=max_value)


def arrays(dims):
    return hypothesis.extra.numpy.arrays(
        np.float32, dims,
        elements=st.floats(min_value=-1.0, max_value=1.0))


def tensor(min_dim=1, max_dim=4, **kwargs):
    dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
    return dims_.flatmap(lambda dims: arrays(dims))


def tensors(n, min_dim=1, max_dim=4, **kwargs):
    dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
    return dims_.flatmap(
        lambda dims: st.lists(arrays(dims), min_size=n, max_size=n))


cpu_do = caffe2_pb2.DeviceOption()
gpu_do = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA)
device_options = [cpu_do] + ([gpu_do] if workspace.has_gpu_support else [])


def device_checker_device_options():
    return st.just(device_options)


def gradient_checker_device_option():
    return st.sampled_from(device_options)

gcs = dict(gc=gradient_checker_device_option(),
           dc=device_checker_device_options())


@contextlib.contextmanager
def temp_workspace(name=b"temp_ws"):
    old_ws_name = workspace.CurrentWorkspace()
    workspace.SwitchWorkspace(name, True)
    yield
    workspace.ResetWorkspace()
    workspace.SwitchWorkspace(old_ws_name)


class HypothesisTestCase(test_util.TestCase):
    def assertDeviceChecks(self, device_options, op, inputs, outputs_to_check,
                           input_device_options=None, threshold=0.01):
        dc = device_checker.DeviceChecker(threshold,
                                          device_options=device_options)
        self.assertTrue(
            dc.CheckSimple(op, inputs, outputs_to_check, input_device_options))

    def assertGradientChecks(self, device_option, op, inputs, outputs_to_check,
                             grad_ops=None, threshold=0.005, stepsize=0.05):
        gc = gradient_checker.GradientChecker(
            stepsize=stepsize, threshold=threshold,
            device_option=device_option,
            workspace_name=str(device_option))
        res, grad, grad_estimated = gc.CheckSimple(
            op, inputs, outputs_to_check, grad_ops)
        self.assertEqual(grad.shape, grad_estimated.shape)
        self.assertTrue(res)

    def assertReferenceChecks(
            self, device_option, op, inputs, reference,
            input_device_options=None, threshold=1e-4):
        op = copy.deepcopy(op)
        op.device_option.CopyFrom(device_option)

        with temp_workspace():
            for (n, b) in zip(op.input, inputs):
                workspace.FeedBlob(
                    n, b,
                    device_option=input_device_options.get(n, device_option))
            workspace.RunOperatorOnce(op)
            reference_outputs = reference(*inputs)
            self.assertEqual(len(reference_outputs), len(op.output))
            for (n, ref) in zip(op.output, reference_outputs):
                output = workspace.FetchBlob(n)
                np.testing.assert_allclose(output, ref, atol=1e-4, rtol=1e-4)
