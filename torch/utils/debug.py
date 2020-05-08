import torch

def is_cpp_stacktraces_enabled():
    r"""Check if the printing of cpp stacktraces is enabled or not.

    """
    return torch._C._is_cpp_stacktraces_enabled()

class cpp_stacktraces(object):
    r"""Context-manager that turn on or off the printing of cpp stacktraces on error.

    While this is enabled, full stack traces from the cpp side will be added to the python
    error message. This is very verbose but is useful for debugging.

    """

    def __init__(self):
        self.prev = torch._C._is_cpp_stacktraces_enabled()

    def __enter__(self):
        torch._C._set_cpp_stacktraces_enabled(True)

    def __exit__(self, *args):
        torch._C._set_cpp_stacktraces_enabled(self.prev)
        return False


class set_cpp_stacktraces(object):
    r"""Context-manager/function that turn on or off the printing of cpp stacktraces on error.

    It can be used as a context-manager or as a function.

    See ``cpp_stacktraces`` above for details.

    Arguments:
        mode (bool): Flag whether to enable cpp stacktraces (``True``),
                     or disable them (``False``).

    """

    def __init__(self, mode):
        self.prev = torch._C._is_cpp_stacktraces_enabled()
        torch._C._set_cpp_stacktraces_enabled(mode)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        torch._C._set_cpp_stacktraces_enabled(self.prev)
        return False
