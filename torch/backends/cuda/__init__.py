import sys
import torch


class ContextProp(object):
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter()

    def __set__(self, obj, val):
        self.setter(val)


class CUDAModule(object):
    def __init__(self, m):
        self.__dict__ = m.__dict__
        # You have to retain the old module, otherwise it will
        # get GC'ed and a lot of things will break.  See:
        # https://stackoverflow.com/questions/47540722/how-do-i-use-the-sys-modules-replacement-trick-in-init-py-on-python-2
        self.__old_mod = m

    cufft_cache_plan = ContextProp(torch._C._get_cufft_cache_plan, torch._C._set_cufft_cache_plan)

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = CUDAModule(sys.modules[__name__])
