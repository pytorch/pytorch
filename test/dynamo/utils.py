# Owner(s): ["module: dynamo"]
import importlib
import os
import sys
import types

import torch
import torch._dynamo


g_tensor_export = torch.ones(10)


tensor_for_import_testing = torch.ones(10, 10)


def inner_func():
    return torch.is_grad_enabled()


def outer_func(func):
    def wrapped(*args):
        a = func(*args)
        torch._dynamo.graph_break()
        return torch.sin(a + 1), inner_func()

    return wrapped


# Create a dummy python module and function to test skipfiles rules.
module_code = """
def add(x):
    return x + 1
"""


def add(x):
    return x + 1


def create_dummy_module_and_function():
    module = types.ModuleType("dummy_module")
    module.__spec__ = importlib.machinery.ModuleSpec(
        "dummy_module", None, origin=os.path.abspath(__file__)
    )
    exec(module_code, module.__dict__)
    sys.modules["dummy_module"] = module
    # Need to override the original function since its __code__.co_filename is not a regular python file name,
    # and the skipfiles rules use filename when checking SKIP_DIRS.
    module.add = add
    return module, module.add
