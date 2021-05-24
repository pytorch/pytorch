import torch

from torch.jit._serialization import validate_map_location

import pathlib
import os

def _load_for_lite_interpreter(f, map_location=None):
    r"""
    Load a :class:`LiteScriptModule`
    saved with :func:`torch.jit._save_for_lite_interpreter`

    Args:
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
    """
    if isinstance(f, str):
        if not os.path.exists(f):
            raise ValueError("The provided filename {} does not exist".format(f))
        if os.path.isdir(f):
            raise ValueError("The provided filename {} is a directory".format(f))

    map_location = validate_map_location(map_location)

    if isinstance(f, str) or isinstance(f, pathlib.Path):
        cpp_module = torch._C._load_for_lite_interpreter(f, map_location)  # type: ignore[attr-defined]
    else:
        cpp_module = torch._C._load_for_lite_interpreter_from_buffer(f.read(), map_location)  # type: ignore[attr-defined]

    return LiteScriptModule(cpp_module)

class LiteScriptModule(object):
    def __init__(self, cpp_module):
        self._c = cpp_module
        super(LiteScriptModule, self).__init__()

    def __call__(self, *input):
        return self._c.forward(input)

    def find_method(self, method_name):
        return self._c.find_method(method_name)

    def forward(self, *input):
        return self._c.forward(input)

    def run_method(self, method_name, *input):
        return self._c.run_method(method_name, input)

def _export_operator_list(module: LiteScriptModule):
    r"""
        return a set of root operator names (with overload name) that are used by any method
        in this mobile module.
    """
    return torch._C._export_operator_list(module._c)

def _get_model_bytecode_version(f_input) -> int:
    r"""
    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name

    Returns:
        version: An integer. If the integer is -1, the version is invalid. A warning
            will show in the log.

    Example:

    .. testcode::

        from torch.jit.mobile import _get_model_bytecode_version

        # Get bytecode version from a saved file path
        version = _get_model_bytecode_version("path/to/model.ptl")

    """
    if isinstance(f_input, str):
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")

    if (isinstance(f_input, str) or isinstance(f_input, pathlib.Path)):
        return torch._C._get_model_bytecode_version(str(f_input))  # type: ignore[attr-defined]
    else:
        return torch._C._get_model_bytecode_version_from_buffer(f_input.read())  # type: ignore[attr-defined]

def _backport_for_mobile(f_input, f_output, to_version):
    r"""
    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        f_output: path to new model destination
        to_version: the expected output model bytecode version
    Returns:
        success: A boolean. If backport success, return true, otherwise false
    """
    if isinstance(f_input, str):
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")

    if ((isinstance(f_input, str) or isinstance(f_input, pathlib.Path)) and (
            isinstance(f_output, str) or isinstance(f_output, pathlib.Path))):
        return torch._C._backport_for_mobile(str(f_input), str(f_output), to_version)  # type: ignore[attr-defined]
    else:
        return torch._C._backport_for_mobile_from_buffer(f_input.read(), str(f_output), to_version)  # type: ignore[attr-defined]

def _backport_for_mobile_to_buffer(f_input, to_version):
    r"""
    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name

    """
    if isinstance(f_input, str):
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")

    if (isinstance(f_input, str) or isinstance(f_input, pathlib.Path)):
        return torch._C._backport_for_mobile_to_buffer(str(f_input), to_version)  # type: ignore[attr-defined]
    else:
        return torch._C._backport_for_mobile_from_buffer_to_buffer(f_input.read(), to_version)  # type: ignore[attr-defined]
