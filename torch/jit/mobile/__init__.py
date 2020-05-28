import torch._C
import torch._jit_internal as _jit_internal
import torch.jit.annotations

from torch.nn import Module

import pathlib
from torch._six import string_classes
from torch.serialization import validate_cuda_device

import os

def load_for_lite_interpreter(f, map_location=None):
    r"""
    Load a :class:`LiteScriptModule`
    saved with :func:`torch.jit._save_for_lite_interpreter <torch.jit._save_for_lite_interpreter>`

    Arguments:
        f: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        map_location: a string or torch.device used to dynamically remap
            storages to an alternative set of devices.

    Returns:
        A :class:`LiteScriptModule` object.

    Example:

    .. testcode::

        import torch
        import io

        # Load LiteScriptModule from saved file path
        torch.jit._load_for_lite_interpreter('lite_script_module.pt')

        # Load LiteScriptModule from io.BytesIO object
        with open('lite_script_module.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())

        # Load all tensors to the original device
        torch.jit.mobile._load_for_lite_interpreter(buffer)

    .. testcleanup::

        import os
        os.remove("lite_script_module.pt")
    """
    print(f)
    if isinstance(f, string_classes):
        if not os.path.exists(f):
            raise ValueError("The provided filename {} does not exist".format(f))
        if os.path.isdir(f):
            raise ValueError("The provided filename {} is a directory".format(f))

    if isinstance(map_location, string_classes):
        map_location = torch.device(map_location)
    elif not (map_location is None or
              isinstance(map_location, torch.device)):
        raise ValueError("map_location should be either None, string or torch.device, "
                         "but got type: " + str(type(map_location)))
    if (str(map_location).startswith('cuda')):
        validate_cuda_device(map_location)

    if isinstance(f, str) or isinstance(f, pathlib.Path):
        cpp_module = torch._C._load_for_lite_interpreter(f, map_location)
    else:
        cpp_module = torch._C._load_for_lite_interpreter_from_buffer(f.read(), map_location)

    return LiteScriptModule(cpp_module)


class LiteScriptModule(object):
    def __init__(self, cpp_module):
        self._c = cpp_module
        super(LiteScriptModule, self).__init__()

    def __call__(self, *input):
        return self._c.forward(input)

    def forward(self, *input):
        return self._c.forward(input)

    def run_method(self, method_name, *input):
        if method_name == '':
            raise KeyError("method name can't be empty string \"\"")
        return self._c.run_method(method_name, input)
