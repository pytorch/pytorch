# Owner(s): ["module: nn"]

import contextlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import NOTEST_CPU, run_tests


# Context manager to set env variable for debug mode
@contextlib.contextmanager
def debug_mode(enable_debug_mode: bool):
    r"""
    This context manager can be used to temporarily enable or disable debug mode
    Upon exiting the context manager, the previous state of the flag will be restored.
    """
    previous_mode: bool = os.environ.get("TORCH_DEBUG", False)
    try:
        os.environ["TORCH_DEBUG"] = str(enable_debug_mode)
        yield {}
    finally:
        os.environ["TORCH_DEBUG"] = str(previous_mode)


class TestDebugMode(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def test_embedding(self, device):
        e = nn.Embedding(10, 10, device=device)
        x = torch.tensor([[10]], device=device)
        error_type = IndexError if device == "cpu" else RuntimeError
        error_message = (
            "index out of range in self"
            if device == "cpu"
            else "CUDA error: device-side assert triggered"
        )
        if device == "cpu":
            # TODO Catching that cuda assert is hard!
            with self.assertRaisesRegex(error_type, error_message):
                e(x)
        with debug_mode(True):
            debug_error_message = "Embedding index out of bounds, value: 10 at index 0, is out of bounds of size : \[10\]"
            with self.assertRaisesRegex(RuntimeError, debug_error_message):
                e(x)


if NOTEST_CPU:
    device_types = ("cuda",)
else:
    device_types = ("cpu", "cuda")

instantiate_device_type_tests(TestDebugMode, globals(), only_for=device_types)


if __name__ == "__main__":
    run_tests()
