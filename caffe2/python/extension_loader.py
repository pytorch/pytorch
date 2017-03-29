## @package extension_loader
# Module caffe2.python.extension_loader
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib

@contextlib.contextmanager
def DlopenGuard():
    # This is a stub for setting up special tricks around python extensions
    # loading. For example, it might do
    #   sys.setdlopenflags(DLFCN.RTLD_GLOBAL | DLFCN.RTLD_NOW)
    # which might be required in some setups of python
    yield
