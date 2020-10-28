## @package extension_loader
# Module caffe2.python.extension_loader




import contextlib
import ctypes
import sys


_set_global_flags = (
    hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'))


@contextlib.contextmanager
def DlopenGuard(extra_flags=ctypes.RTLD_GLOBAL):
    if _set_global_flags:
        old_flags = sys.getdlopenflags()
        sys.setdlopenflags(old_flags | extra_flags)

    # in case we dlopen something that doesn't exist, yield will fail and throw;
    # we need to remember reset the old flags to clean up, otherwise RTLD_GLOBAL
    # flag will stick around and create symbol conflict problems
    try:
        yield
    finally:
        if _set_global_flags:
            sys.setdlopenflags(old_flags)
