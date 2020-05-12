import torch
import os

# On loading, check environment variable for default value
def _set_from_env():
    value = os.getenv("TORCH_CPP_STACKTRACES", "0")
    if value != "0":
        torch._C._set_cpp_stacktraces_enabled(True)
    else:
        torch._C._set_cpp_stacktraces_enabled(False)

_set_from_env()

def is_cpp_stacktraces_enabled():
    r"""Check if the printing of cpp stacktraces is enabled or not.

    """
    return torch._C._is_cpp_stacktraces_enabled()

def set_cpp_stacktraces_enabled(mode):
    r"""Function that turn on or off the printing of cpp stacktraces on error.

    While this is enabled, full stack traces from the cpp side will be added to the python
    error message. This is very verbose but is useful for debugging.

    Arguments:
        mode (bool): Flag whether to enable cpp stacktraces (``True``),
                     or disable them (``False``).

    """
    torch._C._set_cpp_stacktraces_enabled(mode)
