import argparse
import os.path
import sys
import torch


def get_custom_backend_library_path():
    """
    Get the path to the library containing the custom backend.

    Return:
        The path to the custom backend object, customized by platform.
    """
    if sys.platform.startswith("win32"):
        library_filename = "custom_backend.dll"
    elif sys.platform.startswith("darwin"):
        library_filename = "libcustom_backend.dylib"
    else:
        library_filename = "libcustom_backend.so"
    path = os.path.abspath("build/{}".format(library_filename))
    assert os.path.exists(path), path
    return path


def to_custom_backend(module):
    """
    This is a helper that wraps torch._C._jit_to_test_backend and compiles
    only the forward method with an empty compile spec.

    Args:
        module: input ScriptModule.

    Returns:
        The module, lowered so that it can run on TestBackend.
    """
    lowered_module = torch._C._jit_to_backend("custom_backend", module, {"forward": {"": ""}})
    return lowered_module


class Model(torch.nn.Module):
    """
    Simple model used for testing that to_backend API supports saving, loading,
    and executing in C++.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a, b):
        return (a + b, a - b)


def main():
    parser = argparse.ArgumentParser(
        description="Lower a Module to a custom backend"
    )
    parser.add_argument("--export-module-to", required=True)
    options = parser.parse_args()

    # Load the library containing the custom backend.
    library_path = get_custom_backend_library_path()
    torch.ops.load_library(library_path)
    assert library_path in torch.ops.loaded_libraries

    # Lower an instance of Model to the custom backend  and export it
    # to the specified location.
    lowered_module = to_custom_backend(torch.jit.script(Model()))
    torch.jit.save(lowered_module, options.export_module_to)


if __name__ == "__main__":
    main()
