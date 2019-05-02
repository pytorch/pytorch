import sys
import torch


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


class CUDAModule(object):
    def __init__(self, m):
        self.__dict__ = m.__dict__
        # You have to retain the old module, otherwise it will
        # get GC'ed and a lot of things will break.  See:
        # https://stackoverflow.com/questions/47540722/how-do-i-use-the-sys-modules-replacement-trick-in-init-py-on-python-2
        self.__old_mod = m

    cufft_plan_cache = cuFFTPlanCacheManager()

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = CUDAModule(sys.modules[__name__])
