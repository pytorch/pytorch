import sys
import torch

from typing import Union


def is_built():
    r"""Returns whether PyTorch is built with CUDA support.  Note that this
    doesn't necessarily mean CUDA is available; just that if this PyTorch
    binary were run a machine with working CUDA drivers and devices, we
    would be able to use it."""
    return torch._C.has_cuda


class cuFFTPlanCacheAttrContextProp(object):
    # Like regular ContextProp, but uses the `.device_index` attribute from the
    # calling object as the first argument to the getter and setter.
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter(obj.device_index)

    def __set__(self, obj, val):
        if isinstance(self.setter, str):
            raise RuntimeError(self.setter)
        self.setter(obj.device_index, val)


class cuFFTPlanCache(object):
    r"""
    Represents a specific plan cache for a specific `device_index`. The
    attributes `size` and `max_size`, and method `clear`, can fetch and/ or
    change properties of the C++ cuFFT plan cache.
    """
    def __init__(self, device_index):
        self.device_index = device_index

    size = cuFFTPlanCacheAttrContextProp(
        torch._cufft_get_plan_cache_size,
        '.size is a read-only property showing the number of plans currently in the '
        'cache. To change the cache capacity, set cufft_plan_cache.max_size.')

    max_size = cuFFTPlanCacheAttrContextProp(torch._cufft_get_plan_cache_max_size,
                                             torch._cufft_set_plan_cache_max_size)

    def clear(self):
        return torch._cufft_clear_plan_cache(self.device_index)


class cuFFTPlanCacheManager(object):
    r"""
    Represents all cuFFT plan caches. When indexed with a device object/index,
    this object returns the `cuFFTPlanCache` corresponding to that device.

    Finally, this object, when used directly as a `cuFFTPlanCache` object (e.g.,
    setting the `.max_size`) attribute, the current device's cuFFT plan cache is
    used.
    """

    __initialized = False

    def __init__(self):
        self.caches = []
        self.__initialized = True

    def __getitem__(self, device):
        index = torch.cuda._utils._get_device_index(device)
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                ("cufft_plan_cache: expected 0 <= device index < {}, but got "
                 "device with index {}").format(torch.cuda.device_count(), index))
        if len(self.caches) == 0:
            self.caches.extend(cuFFTPlanCache(index) for index in range(torch.cuda.device_count()))
        return self.caches[index]

    def __getattr__(self, name):
        return getattr(self[torch.cuda.current_device()], name)

    def __setattr__(self, name, value):
        if self.__initialized:
            return setattr(self[torch.cuda.current_device()], name, value)
        else:
            return super(cuFFTPlanCacheManager, self).__setattr__(name, value)


class cuBLASModule:
    def __getattr__(self, name):
        if name == "allow_tf32":
            return torch._C._get_cublas_allow_tf32()
        elif name == "allow_fp16_reduced_precision_reduction":
            return torch._C._get_cublas_allow_fp16_reduced_precision_reduction()
        raise AssertionError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            return torch._C._set_cublas_allow_tf32(value)
        elif name == "allow_fp16_reduced_precision_reduction":
            return torch._C._set_cublas_allow_fp16_reduced_precision_reduction(value)
        raise AssertionError("Unknown attribute " + name)

_LinalgBackends = {
    'default': torch._C._LinalgBackend.Default,
    'cusolver': torch._C._LinalgBackend.Cusolver,
    'magma': torch._C._LinalgBackend.Magma,
}
_LinalgBackends_str = ', '.join(_LinalgBackends.keys())

def preferred_linalg_library(backend: Union[None, str, torch._C._LinalgBackend] = None) -> torch._C._LinalgBackend:
    r'''
    .. warning:: This flag is experimental and subject to change.

    When PyTorch runs a CUDA linear algebra operation it often uses the cuSOLVER or MAGMA libraries,
    and if both are available it decides which to use with a heuristic.
    This flag (a :class:`str`) allows overriding those heuristics.

    * If `"cusolver"` is set then cuSOLVER will be used wherever possible.
    * If `"magma"` is set then MAGMA will be used wherever possible.
    * If `"default"` (the default) is set then heuristics will be used to pick between
      cuSOLVER and MAGMA if both are available.
    * When no input is given, this function returns the currently preferred library.

    Note: When a library is preferred other libraries may still be used if the preferred library
    doesn't implement the operation(s) called.
    This flag may achieve better performance if PyTorch's heuristic library selection is incorrect
    for your application's inputs.

    Currently supported linalg operators:

    * :func:`torch.linalg.inv`
    * :func:`torch.linalg.inv_ex`
    * :func:`torch.linalg.cholesky`
    * :func:`torch.linalg.cholesky_ex`
    * :func:`torch.cholesky_solve`
    * :func:`torch.cholesky_inverse`
    * :func:`torch.lu`
    * :func:`torch.linalg.qr`
    * :func:`torch.linalg.eigh`
    * :func:`torch.linalg.svd`
    '''

    if backend is None:
        pass
    elif isinstance(backend, str):
        if backend not in _LinalgBackends:
            raise RuntimeError("Unknown input value. "
                               f"Choose from: {_LinalgBackends_str}.")
        torch._C._set_linalg_preferred_backend(_LinalgBackends[backend])
    elif isinstance(backend, torch._C._LinalgBackend):
        torch._C._set_linalg_preferred_backend(backend)
    else:
        raise RuntimeError("Unknown input value type.")

    return torch._C._get_linalg_preferred_backend()

cufft_plan_cache = cuFFTPlanCacheManager()
matmul = cuBLASModule()
