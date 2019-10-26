import sys
import torch
import warnings
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

try:
    from torch._C import _cudnn
except ImportError:
    _cudnn = None

# Write:
#
#   torch.backends.cudnn.enabled = False
#
# to globally disable CuDNN

__cudnn_version = None

if _cudnn is not None:
    def _init():
        global __cudnn_version
        if __cudnn_version is None:
            __cudnn_version = _cudnn.getVersion()
            compile_version = torch._C._cudnn_version()
            # cuDNN version is MAJOR*1000 + MINOR*100 + PATCH
            runtime_major = __cudnn_version // 1000
            runtime_minor = (__cudnn_version % 1000) // 100
            compile_major = compile_version // 1000
            compile_minor = (compile_version % 1000) // 100
            # Different major versions are always incompatible
            # Starting with cuDNN 7, minor versions are backwards-compatible
            if runtime_major != compile_major:
                cudnn_compatible = False
            elif runtime_major < 7:
                cudnn_compatible = runtime_minor == compile_minor
            else:
                cudnn_compatible = runtime_minor >= compile_minor
            if not cudnn_compatible:
                raise RuntimeError(
                    'cuDNN version incompatibility: PyTorch was compiled against {} '
                    'but linked against {}'.format(compile_version, __cudnn_version))
        return True
else:
    def _init():
        return False


def version():
    if not _init():
        return None
    return __cudnn_version


CUDNN_TENSOR_TYPES = {
    'torch.cuda.HalfTensor',
    'torch.cuda.FloatTensor',
    'torch.cuda.DoubleTensor',
}


def is_available():
    r"""Returns a bool indicating if CUDNN is currently available."""
    return torch._C.has_cudnn


def is_acceptable(tensor):
    if not torch._C._get_cudnn_enabled():
        return False
    if tensor.type() not in CUDNN_TENSOR_TYPES:
        return False
    if not is_available():
        warnings.warn(
            "PyTorch was compiled without cuDNN support. To use cuDNN, rebuild "
            "PyTorch making sure the library is visible to the build system.")
        return False
    if not _init():
        return False
    return True


_handles = {}
verbose = False


def set_flags(_enabled, _benchmark, _deterministic, _verbose):
    global benchmark, deterministic, verbose
    orig_flags = (torch._C._get_cudnn_enabled(),
                  torch._C._get_cudnn_benchmark(),
                  torch._C._get_cudnn_deterministic(),
                  verbose)
    verbose = _verbose
    torch._C._set_cudnn_enabled(_enabled)
    torch._C._set_cudnn_benchmark(_benchmark)
    torch._C._set_cudnn_deterministic(_deterministic)
    return orig_flags


@contextmanager
def flags(enabled=False, benchmark=False, deterministic=False, verbose=False):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, benchmark, deterministic, verbose)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0], orig_flags[1], orig_flags[2], orig_flags[3])


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.<cudnn|mkldnn>.enabled = True

class CudnnModule(PropModule):
    def __init__(self, m, name):
        super(CudnnModule, self).__init__(m, name)

    enabled = ContextProp(torch._C._get_cudnn_enabled, torch._C._set_cudnn_enabled)
    deterministic = ContextProp(torch._C._get_cudnn_deterministic, torch._C._set_cudnn_deterministic)
    benchmark = ContextProp(torch._C._get_cudnn_benchmark, torch._C._set_cudnn_benchmark)

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = CudnnModule(sys.modules[__name__], __name__)
