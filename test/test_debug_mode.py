# Owner(s): ["module: nn"]

import contextlib

import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import TestCase
from torch.testing._internal.common_utils import NOTEST_CPU, run_tests


# Context manager to set env variable for debug mode
@contextlib.contextmanager
def debug_mode(enable_debug_mode: bool):
    r"""
    This context manager can be used to temporarily enable or disable debug mode
    Upon exiting the context manager, the previous state of the flag will be restored.
    """
    previous_mode: bool = torch._C.get_runtime_debug()
    try:
        torch._C.set_runtime_debug(enable_debug_mode)
        print(f"set mode to {torch._C.get_runtime_debug()}")
        yield {}
    finally:
        torch._C.set_runtime_debug(previous_mode)


class TestDebugMode(TestCase):
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
            debug_error_message = (
                r"IndexError embedding index out of bounds, "
                r"value: 10 at index 0, is out of bounds of size : \[10\]"
            )
            with self.assertRaisesRegex(RuntimeError, debug_error_message):
                e(x)


if NOTEST_CPU:
    device_types = ("cuda",)
else:
    device_types = ("cpu", "cuda")

instantiate_device_type_tests(TestDebugMode, globals(), only_for=device_types)


if __name__ == "__main__":
    run_tests()
