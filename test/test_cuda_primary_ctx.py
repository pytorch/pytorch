import ctypes
import torch
from common_utils import TestCase, run_tests, skipIfRocm
import unittest
import glob
import os

# NOTE: this needs to be run in a brand new process

# We cannot import TEST_CUDA and TEST_MULTIGPU from common_cuda here,
# because if we do that, the TEST_CUDNN line from common_cuda will be executed
# multiple times as well during the execution of this test suite, and it will
# cause CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2

if not TEST_CUDA:
    print('CUDA not available, skipping tests')
    TestCase = object  # noqa: F811


_caffe2_nvrtc = None


def get_is_primary_context_created(device):
    flags = ctypes.cast((ctypes.c_uint * 1)(), ctypes.POINTER(ctypes.c_uint))
    active = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    global _caffe2_nvrtc
    if _caffe2_nvrtc is None:
        path = glob.glob('{}/lib/libcaffe2_nvrtc.*'.format(os.path.dirname(torch.__file__)))[0]
        _caffe2_nvrtc = ctypes.cdll.LoadLibrary(path)
    result = _caffe2_nvrtc.cuDevicePrimaryCtxGetState(ctypes.c_int(device), flags, active)
    assert result == 0, 'cuDevicePrimaryCtxGetState failed'
    return bool(active[0])


class TestCudaPrimaryCtx(TestCase):
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @skipIfRocm
    def test_cuda_primary_ctx(self):
        # Ensure context has not been created beforehand
        self.assertFalse(get_is_primary_context_created(0))
        self.assertFalse(get_is_primary_context_created(1))

        x = torch.randn(1, device='cuda:1')

        # We should have only created context on 'cuda:1'
        self.assertFalse(get_is_primary_context_created(0))
        self.assertTrue(get_is_primary_context_created(1))

        print(x)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(get_is_primary_context_created(0))
        self.assertTrue(get_is_primary_context_created(1))

        y = torch.randn(1, device='cpu')
        y.copy_(x)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(get_is_primary_context_created(0))
        self.assertTrue(get_is_primary_context_created(1))

    # DO NOT ADD ANY OTHER TESTS HERE!  ABOVE TEST REQUIRES FRESH PROCESS

if __name__ == '__main__':
    run_tests()
