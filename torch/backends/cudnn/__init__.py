import sys
import torch
import warnings
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

try:
    from torch._C import _cudnn
except ImportError:
    _cudnn = None  # type: ignore

# Write:
#
#   torch.backends.cudnn.enabled = False
#
# to globally disable CuDNN/MIOpen

__cudnn_version = None

if _cudnn is not None:
    def _init():
        global __cudnn_version
        if __cudnn_version is None:
            __cudnn_version = _cudnn.getVersionInt()
            runtime_version = _cudnn.getRuntimeVersion()
            compile_version = _cudnn.getCompileVersion()
            runtime_major, runtime_minor, _ = runtime_version
            compile_major, compile_minor, _ = compile_version
            # Different major versions are always incompatible
            # Starting with cuDNN 7, minor versions are backwards-compatible
            # Not sure about MIOpen (ROCm), so always do a strict check
            if runtime_major != compile_major:
                cudnn_compatible = False
            elif runtime_major < 7 or not _cudnn.is_cuda:
                cudnn_compatible = runtime_minor == compile_minor
            else:
                cudnn_compatible = runtime_minor >= compile_minor
            if not cudnn_compatible:
                raise RuntimeError(
                    'cuDNN version incompatibility: PyTorch was compiled against {} '
                    'but linked against {}'.format(compile_version, runtime_version))
        return True
else:
    def _init():
        return False


def version():
    """Returns the version of cuDNN"""
    if not _init():
        return None
    return __cudnn_version


CUDNN_TENSOR_DTYPES = {
    torch.half,
    torch.float,
    torch.double,
}


def is_available():
    r"""Returns a bool indicating if CUDNN is currently available."""
    return torch._C.has_cudnn


def is_acceptable(tensor):
    if not torch._C._get_cudnn_enabled():
        return False
    if tensor.device.type != 'cuda' or tensor.dtype not in CUDNN_TENSOR_DTYPES:
        return False
    if not is_available():
        warnings.warn(
            "PyTorch was compiled without cuDNN/MIOpen support. To use cuDNN/MIOpen, rebuild "
            "PyTorch making sure the library is visible to the build system.")
        return False
    if not _init():
        warnings.warn('cuDNN/MIOpen library not found. Check your {libpath}'.format(
            libpath={
                'darwin': 'DYLD_LIBRARY_PATH',
                'win32': 'PATH'
            }.get(sys.platform, 'LD_LIBRARY_PATH')))
        return False
    return True


def set_flags(_enabled=None, _benchmark=None, _deterministic=None, _allow_tf32=None):
    orig_flags = (torch._C._get_cudnn_enabled(),
                  torch._C._get_cudnn_benchmark(),
                  torch._C._get_cudnn_deterministic(),
                  torch._C._get_cudnn_allow_tf32())
    if _enabled is not None:
        torch._C._set_cudnn_enabled(_enabled)
    if _benchmark is not None:
        torch._C._set_cudnn_benchmark(_benchmark)
    if _deterministic is not None:
        torch._C._set_cudnn_deterministic(_deterministic)
    if _allow_tf32 is not None:
        torch._C._set_cudnn_allow_tf32(_allow_tf32)
    return orig_flags


@contextmanager
def flags(enabled=False, benchmark=False, deterministic=False, allow_tf32=True):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, benchmark, deterministic, allow_tf32)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.<cudnn|mkldnn>.enabled = True

class CudnnModule(PropModule):
    def __init__(self, m, name):
        super(CudnnModule, self).__init__(m, name)

    enabled = ContextProp(torch._C._get_cudnn_enabled, torch._C._set_cudnn_enabled)
    deterministic = ContextProp(torch._C._get_cudnn_deterministic, torch._C._set_cudnn_deterministic)
    benchmark = ContextProp(torch._C._get_cudnn_benchmark, torch._C._set_cudnn_benchmark)
    allow_tf32 = ContextProp(torch._C._get_cudnn_allow_tf32, torch._C._set_cudnn_allow_tf32)

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = CudnnModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
enabled: bool
deterministic: bool
benchmark: bool
