#!/usr/bin/python3
# mypy: allow-untyped-defs
import importlib.abc
import importlib.util
import sys

import torch
from torch.distributed.nn.jit.templates.remote_module_template import (
    get_remote_module_template,
)


_FILE_PREFIX = "_remote_module_"


def get_arg_return_types_from_interface(module_interface):
    assert getattr(module_interface, "__torch_script_interface__", False), (
        "Expect a TorchScript class interface decorated by @torch.jit.interface."
    )
    qualified_name = torch._jit_internal._qualified_name(module_interface)
    cu = torch.jit._state._python_cu
    module_interface_c = cu.get_interface(qualified_name)
    assert "forward" in module_interface_c.getMethodNames(), (
        f"Expect forward in interface methods, while it has {module_interface_c.getMethodNames()}"
    )
    method_schema = module_interface_c.getMethod("forward")

    arg_str_list = []
    arg_type_str_list = []
    assert method_schema is not None
    for argument in method_schema.arguments:
        arg_str_list.append(argument.name)

        if argument.has_default_value():
            default_value_str = f" = {argument.default_value}"
        else:
            default_value_str = ""
        arg_type_str = f"{argument.name}: {argument.type}{default_value_str}"
        arg_type_str_list.append(arg_type_str)

    arg_str_list = arg_str_list[1:]  # Remove "self".
    args_str = ", ".join(arg_str_list)

    arg_type_str_list = arg_type_str_list[1:]  # Remove "self".
    arg_types_str = ", ".join(arg_type_str_list)

    assert len(method_schema.returns) == 1
    argument = method_schema.returns[0]
    return_type_str = str(argument.type)

    return args_str, arg_types_str, return_type_str


class _StringLoader(importlib.abc.SourceLoader):
    """
    A custom loader for dynamically generated Python source code.

    Inherits from SourceLoader for API compatibility but overrides exec_module()
    to avoid bytecode caching issues. The default SourceLoader.exec_module() calls
    cache_from_source() which fails with IndexError when the filename doesn't
    correspond to a real filesystem path with a .py extension.
    """

    def __init__(self, data: str) -> None:
        self.data = data

    def get_source(self, fullname: str) -> str:
        return self.data

    def get_data(self, path: str) -> bytes:
        return self.data.encode("utf-8")

    def get_filename(self, fullname: str) -> str:
        return f"<{fullname}>.py"

    def path_stats(self, path: str) -> dict:
        # Raise OSError since source is dynamically generated (no filesystem stats)
        raise OSError("dynamically generated module has no filesystem stats")

    def exec_module(self, module) -> None:
        """
        Execute the module by compiling and running the source directly.

        This overrides SourceLoader.exec_module() to bypass the problematic
        get_code() -> cache_from_source() code path that fails on dynamic modules.
        """
        source = self.get_source(module.__name__)
        filename = self.get_filename(module.__name__)
        code = compile(source, filename, "exec", dont_inherit=True)
        exec(code, module.__dict__)


def _do_instantiate_remote_module_template(
    generated_module_name, str_dict, enable_moving_cpu_tensors_to_cuda
):
    if generated_module_name in sys.modules:
        return sys.modules[generated_module_name]

    loader = _StringLoader(
        get_remote_module_template(enable_moving_cpu_tensors_to_cuda).format(**str_dict)
    )
    spec = importlib.util.spec_from_loader(
        generated_module_name, loader, origin="torch-git"
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[generated_module_name] = module
    loader.exec_module(module)
    return module


def instantiate_scriptable_remote_module_template(
    module_interface_cls, enable_moving_cpu_tensors_to_cuda=True
):
    if not getattr(module_interface_cls, "__torch_script_interface__", False):
        raise ValueError(
            f"module_interface_cls {module_interface_cls} must be a type object decorated by "
            "@torch.jit.interface"
        )

    # Generate the template instance name.
    module_interface_cls_name = torch._jit_internal._qualified_name(
        module_interface_cls
    ).replace(".", "_")
    generated_module_name = f"{_FILE_PREFIX}{module_interface_cls_name}"

    # Generate type annotation strs.
    assign_module_interface_cls_str = (
        f"from {module_interface_cls.__module__} import "
        f"{module_interface_cls.__name__} as module_interface_cls"
    )
    args_str, arg_types_str, return_type_str = get_arg_return_types_from_interface(
        module_interface_cls
    )
    kwargs_str = ""
    arrow_and_return_type_str = f" -> {return_type_str}"
    arrow_and_future_return_type_str = f" -> Future[{return_type_str}]"

    str_dict = dict(
        assign_module_interface_cls=assign_module_interface_cls_str,
        arg_types=arg_types_str,
        arrow_and_return_type=arrow_and_return_type_str,
        arrow_and_future_return_type=arrow_and_future_return_type_str,
        args=args_str,
        kwargs=kwargs_str,
        jit_script_decorator="@torch.jit.script",
    )
    return _do_instantiate_remote_module_template(
        generated_module_name, str_dict, enable_moving_cpu_tensors_to_cuda
    )


def instantiate_non_scriptable_remote_module_template():
    generated_module_name = f"{_FILE_PREFIX}non_scriptable"
    str_dict = dict(
        assign_module_interface_cls="module_interface_cls = None",
        args="*args",
        kwargs="**kwargs",
        arg_types="*args, **kwargs",
        arrow_and_return_type="",
        arrow_and_future_return_type="",
        jit_script_decorator="",
    )
    # For a non-scriptable template, always enable moving CPU tensors to a cuda device,
    # because there is no syntax limitation on the extra handling caused by the script.
    return _do_instantiate_remote_module_template(generated_module_name, str_dict, True)
