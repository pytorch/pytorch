# mypy: ignore-errors

r"""This file is allowed to initialize CUDA context when imported."""

import functools
import torch
import torch.cuda
from torch.testing._internal.common_utils import LazyVal, TEST_NUMBA, TEST_WITH_ROCM, TEST_CUDA, IS_WINDOWS, IS_MACOS
import inspect
import contextlib
import os
import unittest


CUDA_ALREADY_INITIALIZED_ON_IMPORT = torch.cuda.is_initialized()


TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
CUDA_DEVICE = torch.device("cuda:0") if TEST_CUDA else None
# note: if ROCm is targeted, TEST_CUDNN is code for TEST_MIOPEN
if TEST_WITH_ROCM:
    TEST_CUDNN = LazyVal(lambda: TEST_CUDA)
else:
    TEST_CUDNN = LazyVal(lambda: TEST_CUDA and torch.backends.cudnn.is_acceptable(torch.tensor(1., device=CUDA_DEVICE)))

TEST_CUDNN_VERSION = LazyVal(lambda: torch.backends.cudnn.version() if TEST_CUDNN else 0)

SM53OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (5, 3))
SM60OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (6, 0))
SM70OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0))
SM75OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5))
SM80OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0))
SM89OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9))
SM90OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0))
SM100OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0))
SM120OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (12, 0))

IS_THOR = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
                  and torch.cuda.get_device_capability()[1] > 0)
IS_JETSON = LazyVal(lambda: torch.cuda.is_available() and (torch.cuda.get_device_capability() in [(7, 2), (8, 7)] or IS_THOR))
IS_SM89 = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9))

def evaluate_gfx_arch_within(arch_list):
    if not torch.cuda.is_available():
        return False
    gcn_arch_name = torch.cuda.get_device_properties('cuda').gcnArchName
    effective_arch = os.environ.get('PYTORCH_DEBUG_FLASH_ATTENTION_GCN_ARCH_OVERRIDE', gcn_arch_name)
    # gcnArchName can be complicated strings like gfx90a:sramecc+:xnack-
    # Hence the matching should be done reversely
    return any(arch in effective_arch for arch in arch_list)

def CDNA3OrLater():
    return evaluate_gfx_arch_within(["gfx940", "gfx941", "gfx942", "gfx950"])

def CDNA2OrLater():
    return evaluate_gfx_arch_within(["gfx90a", "gfx942"])

def evaluate_platform_supports_flash_attention():
    if TEST_WITH_ROCM:
        arch_list = ["gfx90a", "gfx942", "gfx1100", "gfx1201", "gfx950"]
        if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0") != "0":
            arch_list += ["gfx1101", "gfx1150", "gfx1151", "gfx1200"]
        return evaluate_gfx_arch_within(arch_list)
    if TEST_CUDA:
        return not IS_WINDOWS and SM80OrLater
    return False

def evaluate_platform_supports_efficient_attention():
    if TEST_WITH_ROCM:
        arch_list = ["gfx90a", "gfx942", "gfx1100", "gfx1201", "gfx950"]
        if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0") != "0":
            arch_list += ["gfx1101", "gfx1150", "gfx1151", "gfx1200"]
        return evaluate_gfx_arch_within(arch_list)
    if TEST_CUDA:
        return True
    return False

def evaluate_platform_supports_cudnn_attention():
    return (not TEST_WITH_ROCM) and SM80OrLater and (TEST_CUDNN_VERSION >= 90000)

PLATFORM_SUPPORTS_FLASH_ATTENTION: bool = LazyVal(lambda: evaluate_platform_supports_flash_attention())
PLATFORM_SUPPORTS_MEM_EFF_ATTENTION: bool = LazyVal(lambda: evaluate_platform_supports_efficient_attention())
PLATFORM_SUPPORTS_CUDNN_ATTENTION: bool = LazyVal(lambda: evaluate_platform_supports_cudnn_attention())
# This condition always evaluates to PLATFORM_SUPPORTS_MEM_EFF_ATTENTION but for logical clarity we keep it separate
PLATFORM_SUPPORTS_FUSED_ATTENTION: bool = LazyVal(lambda: PLATFORM_SUPPORTS_FLASH_ATTENTION or
                                                  PLATFORM_SUPPORTS_CUDNN_ATTENTION or
                                                  PLATFORM_SUPPORTS_MEM_EFF_ATTENTION)

PLATFORM_SUPPORTS_FUSED_SDPA: bool = TEST_CUDA and not TEST_WITH_ROCM

PLATFORM_SUPPORTS_BF16: bool = LazyVal(lambda: TEST_CUDA and SM80OrLater)

def evaluate_platform_supports_fp8():
    if torch.cuda.is_available():
        if torch.version.hip:
            ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split('.')[:2])
            archs = ['gfx94']
            if ROCM_VERSION >= (6, 3):
                archs.extend(['gfx120'])
            if ROCM_VERSION >= (6, 5):
                archs.append('gfx95')
            for arch in archs:
                if arch in torch.cuda.get_device_properties(0).gcnArchName:
                    return True
        else:
            return SM90OrLater or torch.cuda.get_device_capability() == (8, 9)
    return False

PLATFORM_SUPPORTS_FP8: bool = LazyVal(lambda: evaluate_platform_supports_fp8())

PLATFORM_SUPPORTS_MX_GEMM: bool = LazyVal(lambda: TEST_CUDA and SM100OrLater)

if TEST_NUMBA:
    try:
        import numba.cuda
        TEST_NUMBA_CUDA = numba.cuda.is_available()
    except Exception:
        TEST_NUMBA_CUDA = False
        TEST_NUMBA = False
else:
    TEST_NUMBA_CUDA = False

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
            torch.randn(1, device=f"cuda:{i}")
        __cuda_ctx_rng_initialized = True


@contextlib.contextmanager
def tf32_off():
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul


@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-5):
    if torch.version.hip:
        hip_allow_tf32 = os.environ.get("HIPBLASLT_ALLOW_TF32", None)
        os.environ["HIPBLASLT_ALLOW_TF32"] = "1"
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    old_precision = self.precision
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        self.precision = tf32_precision
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=True):
            yield
    finally:
        if torch.version.hip:
            if hip_allow_tf32 is not None:
                os.environ["HIPBLASLT_ALLOW_TF32"] = hip_allow_tf32
            else:
                del os.environ["HIPBLASLT_ALLOW_TF32"]
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul
        self.precision = old_precision


@contextlib.contextmanager
def tf32_enabled():
    """
    Context manager to temporarily enable TF32 for CUDA operations.
    Restores the previous TF32 state after exiting the context.
    """
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=True
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul


# This is a wrapper that wraps a test to run this test twice, one with
# allow_tf32=True, another with allow_tf32=False. When running with
# allow_tf32=True, it will use reduced precision as specified by the
# argument. For example:
#    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
#    @tf32_on_and_off(0.005)
#    def test_matmul(self, device, dtype):
#        a = ...; b = ...;
#        c = torch.matmul(a, b)
#        self.assertEqual(c, expected)
# In the above example, when testing torch.float32 and torch.complex64 on CUDA
# on a CUDA >= 11 build on an >=Ampere architecture, the matmul will be running at
# TF32 mode and TF32 mode off, and on TF32 mode, the assertEqual will use reduced
# precision to check values.
#
# This decorator can be used for function with or without device/dtype, such as
# @tf32_on_and_off(0.005)
# def test_my_op(self)
# @tf32_on_and_off(0.005)
# def test_my_op(self, device)
# @tf32_on_and_off(0.005)
# def test_my_op(self, device, dtype)
# @tf32_on_and_off(0.005)
# def test_my_op(self, dtype)
# if neither device nor dtype is specified, it will check if the system has ampere device
# if device is specified, it will check if device is cuda
# if dtype is specified, it will check if dtype is float32 or complex64
# tf32 and fp32 are different only when all the three checks pass
def tf32_on_and_off(tf32_precision=1e-5):
    def with_tf32_disabled(self, function_call):
        with tf32_off():
            function_call()

    def with_tf32_enabled(self, function_call):
        with tf32_on(self, tf32_precision):
            function_call()

    def wrapper(f):
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            kwargs.update(zip(arg_names, args))
            cond = torch.cuda.is_tf32_supported()
            if 'device' in kwargs:
                cond = cond and (torch.device(kwargs['device']).type == 'cuda')
            if 'dtype' in kwargs:
                cond = cond and (kwargs['dtype'] in {torch.float32, torch.complex64})
            if cond:
                with_tf32_disabled(kwargs['self'], lambda: f(**kwargs))
                with_tf32_enabled(kwargs['self'], lambda: f(**kwargs))
            else:
                f(**kwargs)

        return wrapped
    return wrapper


# This is a wrapper that wraps a test to run it with TF32 turned off.
# This wrapper is designed to be used when a test uses matmul or convolutions
# but the purpose of that test is not testing matmul or convolutions.
# Disabling TF32 will enforce torch.float tensors to be always computed
# at full precision.
def with_tf32_off(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        with tf32_off():
            return f(*args, **kwargs)

    return wrapped

def _get_magma_version():
    if 'Magma' not in torch.__config__.show():
        return (0, 0)
    position = torch.__config__.show().find('Magma ')
    version_str = torch.__config__.show()[position + len('Magma '):].split('\n')[0]
    return tuple(int(x) for x in version_str.split("."))

def _get_torch_cuda_version():
    if torch.version.cuda is None:
        return (0, 0)
    cuda_version = str(torch.version.cuda)
    return tuple(int(x) for x in cuda_version.split("."))

def _get_torch_rocm_version():
    if not TEST_WITH_ROCM or torch.version.hip is None:
        return (0, 0)
    rocm_version = str(torch.version.hip)
    rocm_version = rocm_version.split("-")[0]    # ignore git sha
    return tuple(int(x) for x in rocm_version.split("."))

def _check_cusparse_generic_available():
    return not TEST_WITH_ROCM

def _check_hipsparse_generic_available():
    if not TEST_WITH_ROCM:
        return False
    if not torch.version.hip:
        return False

    rocm_version = str(torch.version.hip)
    rocm_version = rocm_version.split("-")[0]    # ignore git sha
    rocm_version_tuple = tuple(int(x) for x in rocm_version.split("."))
    return not (rocm_version_tuple is None or rocm_version_tuple < (5, 1))


TEST_CUSPARSE_GENERIC = _check_cusparse_generic_available()
TEST_HIPSPARSE_GENERIC = _check_hipsparse_generic_available()

# Shared by test_torch.py and test_multigpu.py
def _create_scaling_models_optimizers(device="cuda", optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
    # Create a module+optimizer that will use scaling, and a control module+optimizer
    # that will not use scaling, against which the scaling-enabled module+optimizer can be compared.
    mod_control = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
    mod_scaling = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
    with torch.no_grad():
        for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
            s.copy_(c)

    kwargs = {"lr": 1.0}
    if optimizer_kwargs is not None:
        kwargs.update(optimizer_kwargs)
    opt_control = optimizer_ctor(mod_control.parameters(), **kwargs)
    opt_scaling = optimizer_ctor(mod_scaling.parameters(), **kwargs)

    return mod_control, mod_scaling, opt_control, opt_scaling

# Shared by test_torch.py, test_cuda.py and test_multigpu.py
def _create_scaling_case(device="cuda", dtype=torch.float, optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
    data = [(torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device))]

    loss_fn = torch.nn.MSELoss().to(device)

    skip_iter = 2

    return _create_scaling_models_optimizers(
        device=device, optimizer_ctor=optimizer_ctor, optimizer_kwargs=optimizer_kwargs,
    ) + (data, loss_fn, skip_iter)


def xfailIfSM89(func):
    return func if not IS_SM89 else unittest.expectedFailure(func)

def xfailIfSM100OrLater(func):
    return func if not SM100OrLater else unittest.expectedFailure(func)

def xfailIfSM120OrLater(func):
    return func if not SM120OrLater else unittest.expectedFailure(func)

def xfailIfDistributedNotSupported(func):
    return func if not (IS_MACOS or IS_JETSON) else unittest.expectedFailure(func)

# Importing this module should NOT eagerly initialize CUDA
if not CUDA_ALREADY_INITIALIZED_ON_IMPORT:
    assert not torch.cuda.is_initialized()
