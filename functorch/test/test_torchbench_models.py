"""test.py
Setup and Run hub models.

Make sure to enable an https proxy if necessary, or the setup steps may hang.
"""
# This file shows how to use the benchmark suite from user end.
import gc
import functools
import os
import traceback
import unittest
from unittest.mock import patch
import yaml

import torch
from torchbenchmark import _list_model_paths, ModelTask, get_metadata_from_yaml
from torchbenchmark.util.metadata_utils import skip_by_metadata


# Some of the models have very heavyweight setup, so we have to set a very
# generous limit. That said, we don't want the entire test suite to hang if
# a single test encounters an extreme failure, so we give up after 5 a test
# is unresponsive to 5 minutes. (Note: this does not require that the entire
# test case completes in 5 minutes. It requires that if the worker is
# unresponsive for 5 minutes the parent will presume it dead / incapacitated.)
TIMEOUT = 300  # Seconds

class TestBenchmark(unittest.TestCase):

    def setUp(self):
        gc.collect()

    def tearDown(self):
        gc.collect()

    def test_fx_profile(self):
        try:
            from torch.fx.interpreter import Interpreter
        except ImportError:  # older versions of PyTorch
            raise unittest.SkipTest("Requires torch>=1.8")
        from fx_profile import main, ProfileAggregate
        with patch.object(ProfileAggregate, "save") as mock_save:
            # just run one model to make sure things aren't completely broken
            main(["--repeat=1", "--filter=pytorch_struct", "--device=cpu"])
            self.assertGreaterEqual(mock_save.call_count, 1)

def _create_example_model_instance(task: ModelTask, device: str):
    skip = False
    try:
        task.make_model_instance(test="eval", device=device, jit=False)
    except NotImplementedError:
        try:
            task.make_model_instance(test="train", device=device, jit=False)
        except NotImplementedError:
            skip = True
    finally:
        if skip:
            raise NotImplementedError(f"Model is not implemented on the device {device}")

def _load_test(path, device):

    def example_fn(self):
        task = ModelTask(path, timeout=TIMEOUT)
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                _create_example_model_instance(task, device)
                task.check_example()
                task.del_model_instance()
            except NotImplementedError:
                self.skipTest(f'Method `get_module()` on {device} is not implemented, skipping...')

    def train_fn(self):
        metadata = get_metadata_from_yaml(path)
        task = ModelTask(path, timeout=TIMEOUT)
        allow_customize_batch_size = task.get_model_attribute("ALLOW_CUSTOMIZE_BSIZE", classattr=True)
        # to speedup test, use batch size 1 if possible
        batch_size = 1 if allow_customize_batch_size else None
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                task.make_model_instance(test="train", device=device, jit=False, batch_size=batch_size)
                task.invoke()
                task.check_details_train(device=device, md=metadata)
                task.del_model_instance()
            except NotImplementedError:
                self.skipTest(f'Method train on {device} is not implemented, skipping...')

    def eval_fn(self):
        metadata = get_metadata_from_yaml(path)
        task = ModelTask(path, timeout=TIMEOUT)
        allow_customize_batch_size = task.get_model_attribute("ALLOW_CUSTOMIZE_BSIZE", classattr=True)
        # to speedup test, use batch size 1 if possible
        batch_size = 1 if allow_customize_batch_size else None
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                task.make_model_instance(test="eval", device=device, jit=False, batch_size=batch_size)
                task.invoke()
                task.check_details_eval(device=device, md=metadata)
                task.check_eval_output()
                task.del_model_instance()
            except NotImplementedError:
                self.skipTest(f'Method eval on {device} is not implemented, skipping...')

    def check_device_fn(self):
        task = ModelTask(path, timeout=TIMEOUT)
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                task.make_model_instance(test="eval", device=device, jit=False)
                task.check_device()
                task.del_model_instance()
            except NotImplementedError:
                self.skipTest(f'Method check_device on {device} is not implemented, skipping...')

    def check_functorch(self):
        task = ModelTask(path, timeout=TIMEOUT)
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                task.make_model_instance(test="train", device=device, jit=False)
                task.check_functorch()
                task.del_model_instance()
            except NotImplementedError:
                self.skipTest(f'Method check_device on {device} is not implemented, skipping...')


    name = os.path.basename(path)
    metadata = get_metadata_from_yaml(path)
    for fn, fn_name in zip([check_functorch],
                           ["check_functorch"]):
        # set exclude list based on metadata
        setattr(TestBenchmark, f'test_{name}_{fn_name}_{device}',
                (unittest.skipIf(skip_by_metadata(test=fn_name, device=device,\
                                                  jit=False, extra_args=[], metadata=metadata), "This test is skipped by its metadata")(fn)))


def _load_tests():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')

    for path in _list_model_paths():
        # TODO: skipping quantized tests for now due to BC-breaking changes for prepare
        # api, enable after PyTorch 1.13 release
        if "quantized" in path:
            continue
        for device in devices:
            _load_test(path, device)


_load_tests()
if __name__ == '__main__':
    unittest.main()
