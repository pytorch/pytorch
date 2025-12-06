# mypy: ignore-errors

import contextlib
import functools
import logging
import os
import re
import sys
import unittest
from subprocess import CalledProcessError

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._inductor.codecache import CppCodeCache
from torch._inductor.codegen.common import (
    get_custom_backend_config_for_device,
    get_custom_backend_pass_for_device,
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    init_backend_registration,
    register_backend_for_device,
)
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.compile_fx import shape_env_from_inputs
from torch._inductor.custom_graph_pass import CustomGraphModulePass
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import (
    get_gpu_shared_memory,
    get_gpu_type,
    GPU_TYPES,
    is_big_gpu,
    is_gpu,
    OrderedSet,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._helion import has_helion
from torch.utils._pallas import has_pallas
from torch.utils._triton import has_triton
from torch.utils._config_module import ConfigModule
from torch.testing._internal.common_device_type import (
    get_desired_device_type_test_bases,
)
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_FBCODE,
    IS_WINDOWS,
    LazyVal,
    TestCase,
)

log: logging.Logger = logging.getLogger(__name__)


def test_cpu():
    try:
        CppCodeCache.load("")
        return not IS_FBCODE
    except (
        CalledProcessError,
        OSError,
        torch._inductor.exc.InvalidCxxCompiler,
        torch._inductor.exc.CppCompileError,
    ):
        return False


HAS_CPU = LazyVal(test_cpu)

HAS_TRITON = has_triton()

HAS_PALLAS = has_pallas()

HAS_HELION = has_helion()

if HAS_TRITON:
    import triton

    TRITON_HAS_CPU = "cpu" in triton.backends.backends
else:
    TRITON_HAS_CPU = False


HAS_CUDA_AND_TRITON = torch.cuda.is_available() and HAS_TRITON

HAS_XPU_AND_TRITON = torch.xpu.is_available() and HAS_TRITON

HAS_MPS = torch.mps.is_available()

HAS_GPU = HAS_CUDA_AND_TRITON or HAS_XPU_AND_TRITON
HAS_GPU_AND_TRITON = HAS_GPU

GPU_TYPE = get_gpu_type()

HAS_MULTIGPU = any(
    getattr(torch, gpu).is_available() and getattr(torch, gpu).device_count() >= 2
    for gpu in GPU_TYPES
)

_desired_test_bases = get_desired_device_type_test_bases(allow_xpu=True)
RUN_GPU = HAS_GPU and any(
    is_gpu(getattr(x, "device_type", "")) for x in _desired_test_bases
)

RUN_CPU = HAS_CPU and any(
    getattr(x, "device_type", "") == "cpu" for x in _desired_test_bases
)


def _check_has_dynamic_shape(
    self: TestCase,
    code,
):
    for_loop_found = False
    has_dynamic = False
    lines = code.split("\n")
    for line in lines:
        if "for(" in line:
            for_loop_found = True
            if re.search(r";.*ks.*;", line) is not None:
                has_dynamic = True
                break
    self.assertTrue(
        has_dynamic, msg=f"Failed to find dynamic for loop variable\n{code}"
    )
    self.assertTrue(for_loop_found, f"Failed to find for loop\n{code}")


def skipDeviceIf(cond, msg, *, device):
    if cond:

        def decorate_fn(fn):
            @functools.wraps(fn)
            def inner(self, *args, **kwargs):
                if not hasattr(self, "device"):
                    warn_msg = (
                        "Expect the test class to have attribute device but not found. "
                    )
                    if hasattr(self, "device_type"):
                        warn_msg += "Consider using the skip device decorators in common_device_type.py"
                    log.warning(warn_msg)
                if self.device == device:
                    raise unittest.SkipTest(msg)
                return fn(self, *args, **kwargs)

            return inner

    else:

        def decorate_fn(fn):
            return fn

    return decorate_fn


def skip_windows_ci(name: str, file: str) -> None:
    if IS_WINDOWS and IS_CI:
        module = os.path.basename(file).strip(".py")
        sys.stderr.write(
            f"Windows CI does not have necessary dependencies for {module} tests yet\n"
        )
        if name == "__main__":
            sys.exit(0)
        raise unittest.SkipTest("requires sympy/functorch/filelock")


# TODO: Remove HAS_MPS condition  when `HAS_GPU` includes HAS_MPS
requires_gpu = functools.partial(
    unittest.skipIf, not (HAS_GPU or HAS_MPS), "requires gpu"
)
requires_triton = functools.partial(unittest.skipIf, not HAS_TRITON, "requires triton")
requires_helion = functools.partial(unittest.skipIf, not HAS_HELION, "requires helion")


def requires_cuda_with_enough_memory(min_mem_required):
    def inner(fn):
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_properties().total_memory < min_mem_required
        ):
            return unittest.skip(
                f"Only if the CUDA device has at least {min_mem_required / 1e9:.3f}GB memory to be safe"
            )(fn)
        else:
            return fn

    return inner


skipCUDAIf = functools.partial(skipDeviceIf, device="cuda")
skipXPUIf = functools.partial(skipDeviceIf, device="xpu")
skipCPUIf = functools.partial(skipDeviceIf, device="cpu")

IS_A100 = LazyVal(lambda: HAS_CUDA_AND_TRITON and get_gpu_shared_memory() == 166912)

IS_H100 = LazyVal(lambda: HAS_CUDA_AND_TRITON and get_gpu_shared_memory() == 232448)

IS_BIG_GPU = LazyVal(lambda: HAS_GPU_AND_TRITON and is_big_gpu())


def dummy_graph() -> GraphLowering:
    """
    Create a graph. This is useful for unit testing code which accesses
    V.graph.sizevars.
    """
    example_inputs = [torch.randn(10) for _ in range(2)]
    gm = make_fx(torch.add, tracing_mode="fake")(*example_inputs)
    shape_env = shape_env_from_inputs(example_inputs)
    graph = GraphLowering(
        gm,
        shape_env=shape_env,
    )

    return graph


def maybe_skip_size_asserts(op):
    """
    For certain ops, there meta and eager implementation returns different
    strides. This cause size/strides assert fail. Skip adding those
    asserts for now.
    """
    if (
        op.aten_name
        in (
            "fft_hfftn",
            "fft_hfft",
            "fft_hfft2",
            "fft_ihfftn",
            "fft_fft",
            "fft_fft2",
            "fft_fftn",
            "fft_ifft",
            "fft_ifft2",
            "fft_ifftn",
            "fft_irfft",
            "fft_irfft2",
            "fft_irfftn",
            "fft_ihfft",
            "fft_ihfft2",
            "fft_rfft",
            "fft_rfft2",
            "fft_rfftn",
            "linalg_eig",
            "linalg_eigvals",
        )
        and "TORCHINDUCTOR_SIZE_ASSERTS" not in os.environ
    ):
        return torch._inductor.config.patch(size_asserts=False)
    else:
        return contextlib.nullcontext()


def get_func_call() -> str:
    return (
        "void inductor_entry_impl("
        if torch._inductor.config.cpp_wrapper
        else "def call("
    )


def get_kernel_launch() -> str:
    return "call_triton_" if torch._inductor.config.cpp_wrapper else ".run("


def clone_preserve_strides_offset(x, device=None):
    if not isinstance(x, torch.Tensor):
        return x
    buffer = torch.as_strided(
        x, (x.untyped_storage().size() // x.element_size(),), (1,), 0
    )
    if not device:
        buffer = buffer.clone()
    else:
        buffer = buffer.to(device, copy=True)
    out = torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
    return out


# define the e4m3/e5m2 constants
E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
E4M3FNUZ_MAX_POS = torch.finfo(torch.float8_e4m3fnuz).max
E5M2FNUZ_MAX_POS = torch.finfo(torch.float8_e5m2fnuz).max

FP16_MAX_POS: float = torch.finfo(torch.float16).max
EPS: float = 1e-12

Tensor = torch.Tensor


def _to_fp8_saturated(x: Tensor, float8_dtype: torch.dtype) -> Tensor:
    # The default behavior in PyTorch for casting to `float8_e4m3fn`
    # and `e5m2` is to not saturate. In this context, we should saturate.
    # A common case where we want to saturate is when the history of a
    # tensor has a maximum value of `amax1`, and the current amax value
    # is `amax2`, where `amax1 < amax2`. This is common when using delayed
    # scaling.
    if float8_dtype == torch.float8_e4m3fn:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    elif float8_dtype == torch.float8_e5m2:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    elif float8_dtype == torch.float8_e4m3fnuz:
        x = x.clamp(min=-1 * E4M3FNUZ_MAX_POS, max=E4M3FNUZ_MAX_POS)
    elif float8_dtype == torch.float8_e5m2fnuz:
        x = x.clamp(min=-1 * E5M2FNUZ_MAX_POS, max=E5M2FNUZ_MAX_POS)
    else:
        raise TypeError(f"Unsupported float8_dtype: {float8_dtype}")
    return x.to(float8_dtype)


@torch.no_grad()
def _amax_to_scale(
    amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
) -> torch.Tensor:
    # To make scale dtype to be fp32 for accuracy
    amax = amax.float()
    if float8_dtype == torch.float8_e4m3fn:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    else:  # e5m2
        res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)

    # Ensure that the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16.
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=FP16_MAX_POS)
    return res


def _quantize_tensorwise(x: Tensor, float8_dtype: torch.dtype):
    amax = torch.max(torch.abs(x))
    scale = _amax_to_scale(amax, float8_dtype, x.dtype)
    x_fp8 = _to_fp8_saturated(x * scale, float8_dtype)
    inverse_scale = scale.reciprocal()
    return x_fp8, inverse_scale


def _quantize_rowwise(x: Tensor, float8_dtype: torch.dtype):
    amax = torch.max(torch.abs(x), dim=1, keepdim=True).values
    scale = _amax_to_scale(amax, float8_dtype, x.dtype)
    x_fp8 = _to_fp8_saturated(x * scale, float8_dtype)
    inverse_scale = scale.reciprocal()
    return x_fp8, inverse_scale


def _quantize_blockwise(
    x: Tensor, float8_dtype: torch.dtype, block_outer: int, block_inner: int
):
    min_outer = min(block_outer, x.shape[0])
    min_inner = min(block_inner, x.shape[1])
    x = x.unflatten(1, (-1, min_inner)).unflatten(0, (-1, min_outer))
    amax = x.abs().amax(dim=[1, 3], keepdim=True).float()
    scale = _amax_to_scale(amax, float8_dtype, x.dtype)
    x = x.flatten(2, 3).flatten(0, 1)
    scale = scale.flatten(2, 3).flatten(0, 1)
    scale_expanded = scale.repeat_interleave(min_outer, dim=0).repeat_interleave(
        min_inner, dim=1
    )
    x_fp8 = _to_fp8_saturated(
        x / scale_expanded,  # Ensures that scaling doesn't cause inf/nan values
        float8_dtype,
    )
    inverse_scale = scale.reciprocal()
    return x_fp8, inverse_scale


class MockGraphHandler(GraphLowering):
    """Minimal mock graph handler for testing virtualized context."""

    def __init__(self, name_to_buffer=None):
        import torch._inductor.sizevars

        self.sizevars = torch._inductor.sizevars.SizeVarAllocator()
        self.name_to_buffer = name_to_buffer or {}
        self.graph_inputs = {}
        self.mutated_buffers = OrderedSet()
        self.removed_buffers = OrderedSet()
        self.constants = {}
        self.scheduler = None

    def get_dtype(self, buffer_name: str) -> torch.dtype:  # noqa: ARG002
        """Return default dtype for any buffer (for testing)."""
        return torch.float32


@contextlib.contextmanager
def patch_inductor_backend(
    device: str,
    python_wrapper_codegen: PythonWrapperCodegen = None,
    custom_pass: CustomGraphModulePass = None,
    custom_backend_config: ConfigModule = None,
):
    """
    Patch the inductor backend for a specific device.
    """
    # Make sure the backend is already registered
    init_backend_registration()

    # Get the original registration parameters
    original_scheduling = get_scheduling_for_device(device)
    original_python_wrapper = get_wrapper_codegen_for_device(device, False)
    original_cpp_wrapper = get_wrapper_codegen_for_device(device, True)
    original_fx_wrapper = get_wrapper_codegen_for_device(device, fx_wrapper=True)
    original_custom_pass = get_custom_backend_pass_for_device(device)
    original_custom_backend_config = get_custom_backend_config_for_device(device)

    try:
        # Register modified backend for the device
        register_backend_for_device(
            device,
            original_scheduling,
            (
                python_wrapper_codegen
                if python_wrapper_codegen is not None
                else original_python_wrapper
            ),
            original_cpp_wrapper,
            original_fx_wrapper,
            custom_pass if custom_pass is not None else original_custom_pass,
            (
                custom_backend_config
                if custom_backend_config is not None
                else original_custom_backend_config
            ),
        )
        yield
    finally:
        # Restore the original backend
        register_backend_for_device(
            device,
            original_scheduling,
            original_python_wrapper,
            original_cpp_wrapper,
            original_fx_wrapper,
            original_custom_pass,
            original_custom_backend_config,
        )
