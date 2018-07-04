## @package extension_loader
# Module caffe2.python.extension_loader
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import ctypes
import sys


_set_global_flags = (
    hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'))


@contextlib.contextmanager
def DlopenGuard():
    if _set_global_flags:
        old_flags = sys.getdlopenflags()
        sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    yield
    if _set_global_flags:
        sys.setdlopenflags(old_flags)
