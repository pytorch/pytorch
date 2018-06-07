r"""This file is allowed to initialize CUDA context when imported."""

import torch
import torch.cuda

class common_cuda(object):
    __instance = None

    def __new__(cls):
        if cls.__instance == None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self):
        if not hasattr(self, 'TEST_CUDA'):
            self.TEST_CUDA = torch.cuda.is_available()
        if not hasattr(self, 'TEST_MULTIGPU'):
            self.TEST_MULTIGPU = self.TEST_CUDA and torch.cuda.device_count() >= 2
        if not hasattr(self, 'CUDA_DEVICE'):
            self.CUDA_DEVICE = self.TEST_CUDA and torch.device("cuda:0")
        if not hasattr(self, 'TEST_CUDNN'):
            self.TEST_CUDNN = self.TEST_CUDA and torch.backends.cudnn.is_acceptable(torch.tensor(1., device=self.CUDA_DEVICE))
        if not hasattr(self, 'TEST_CUDNN_VERSION'):
            self.TEST_CUDNN_VERSION = self.TEST_CUDNN and torch.backends.cudnn.version()


TEST_CUDA = common_cuda().TEST_CUDA
TEST_MULTIGPU = common_cuda().TEST_MULTIGPU
CUDA_DEVICE = common_cuda().CUDA_DEVICE
TEST_CUDNN = common_cuda().TEST_CUDNN
TEST_CUDNN_VERSION = common_cuda().TEST_CUDNN_VERSION


# Used below in `initialize_cuda_context_rng` to ensure that CUDA context and
# RNG have been initialized.
__cuda_ctx_rng_initialized = False


# after this call, CUDA context and RNG must have been initialized on each GPU
def initialize_cuda_context_rng():
    global __cuda_ctx_rng_initialized
    assert TEST_CUDA, 'CUDA must be available when calling initialize_cuda_context_rng'
    if not __cuda_ctx_rng_initialized:
        # initialize cuda context and rng for memory tests
        for i in range(torch.cuda.device_count()):
            torch.randn(1, device="cuda:{}".format(i))
        __cuda_ctx_rng_initialized = True
