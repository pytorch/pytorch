from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import sys
import DLFCN


@contextlib.contextmanager
def DlopenGuard():
    # In python 2.7 required constants are not defined.
    # Thus they are listed explicitly
    flags = sys.getdlopenflags()
    sys.setdlopenflags(DLFCN.RTLD_GLOBAL | DLFCN.RTLD_NOW)
    yield
    sys.setdlopenflags(flags)
