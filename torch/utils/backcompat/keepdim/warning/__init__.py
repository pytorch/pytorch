
import sys
from torch._C import _set_backcompat_keepdim_warn
from torch._C import _get_backcompat_keepdim_warn

class Warning:
    def __init__(self):
        pass

    def set_enabled(self, value):
        _set_backcompat_keepdim_warn(value)

    def get_enabled(self):
        return _get_backcompat_keepdim_warn()

    enabled = property(get_enabled, set_enabled)

sys.modules[__name__] = Warning()
