import importlib


def do_import(module, context):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            f'Could not import {module}. The custom operator library "{context}" '
            f"specified that it must be loaded together with this Python module. "
            f"Please check that you have installed the module."
        ) from ex


def initialize_pyimports_handler(_C):
    ignored_pyimports = _C._initialize_pyimports_handler()
    if len(ignored_pyimports) > 0:
        raise RuntimeError(
            f"A PyTorch custom operator shared library was loaded before "
            f"`import torch`. This is not supported; please `import torch` first "
            f"before loading shared libraries with custom operators. List of "
            f"offending libraries: {ignored_pyimports}"
        )
