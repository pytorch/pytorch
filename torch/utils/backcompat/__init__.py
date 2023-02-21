from torch._C import _set_backcompat_broadcast_warn
from torch._C import _get_backcompat_broadcast_warn
from torch._C import _set_backcompat_keepdim_warn
from torch._C import _get_backcompat_keepdim_warn


class Warning:
    def __init__(self, setter, getter):
        self.setter = setter
        self.getter = getter

    def set_enabled(self, value):
        self.setter(value)

    def get_enabled(self):
        return self.getter()

    enabled = property(get_enabled, set_enabled)

broadcast_warning = Warning(_set_backcompat_broadcast_warn, _get_backcompat_broadcast_warn)
keepdim_warning = Warning(_set_backcompat_keepdim_warn, _get_backcompat_keepdim_warn)
