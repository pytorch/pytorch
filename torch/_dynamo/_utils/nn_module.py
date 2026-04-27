from __future__ import annotations

import dis
import enum
import functools
import importlib
import inspect
import os
import types
from collections import OrderedDict
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._logging import LazyString
from torch.fx._utils import _format_graph_code, lazy_format_graph_code
from torch.utils.hooks import RemovableHandle


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._guards import Source


def fqn(obj: Any) -> str:
    """
    Returns the fully qualified name of the object.
    """
    return f"{obj.__module__}.{obj.__qualname__}"


def import_submodule(mod: types.ModuleType) -> None:
    """
    Ensure all the files in a given submodule are imported
    """
    for filename in sorted(os.listdir(os.path.dirname(cast(str, mod.__file__)))):
        if filename.endswith(".py") and filename[0] != "_":
            importlib.import_module(f"{mod.__name__}.{filename[:-3]}")


def object_has_getattribute(value: Any) -> bool:
    return class_has_getattribute(type(value))


def object_setattr_ignore_descriptor(obj: Any, name: str, value: Any) -> None:
    # https://github.com/python/cpython/blob/3.11/Objects/object.c#L1286-L1335
    d = object.__getattribute__(obj, "__dict__")
    d[name] = value


def class_has_getattribute(cls: type) -> bool:
    try:
        if isinstance(
            inspect.getattr_static(cls, "__getattribute__"),
            types.FunctionType,
        ):
            return True
    except AttributeError:
        pass
    return False


def get_custom_getattr(
    value: Any, ignore_nn_module_getattr: bool = False
) -> Any | None:
    try:
        getattr_fn = inspect.getattr_static(type(value), "__getattr__")
    except AttributeError:
        getattr_fn = None
    if ignore_nn_module_getattr and getattr_fn is torch.nn.Module.__getattr__:
        # ignore this case of getattr
        getattr_fn = None
    return getattr_fn


class TensorStaticReason(enum.Enum):
    PARAMETER = 2
    NOT_TENSOR = 4
    NN_MODULE_PROPERTY = 5


def tensor_static_reason_to_message(reason: TensorStaticReason) -> str:
    if reason == TensorStaticReason.PARAMETER:
        return "mark_dynamic on parameter, parameters are always static today."
    if reason == TensorStaticReason.NOT_TENSOR:
        return "mark_dynamic on a non tensor, how did this happen?"
    if reason == TensorStaticReason.NN_MODULE_PROPERTY:
        return "tensor is static because it is nn module associated."
    raise AssertionError(f"Illegal reason {reason}")


def tensor_always_has_static_shape(
    tensor: torch.Tensor | Any,
    is_tensor: bool,
    tensor_source: Source,
) -> tuple[bool, TensorStaticReason | None]:
    """
    Given a tensor, source, and is_tensor flag, determine if a shape should be static.

    Args:
    tensor - the real tensor to evaluate, parameters force a static shape.
    is_tensor - internal dynamo check, essentially "is_tensor": target_cls is TensorVariable,
    tensors not in a TensorVariable for whatever reason are forced static.

    Returns a tuple, where the first element is the bool of whether or not this tensor should have a static shape.
    The second element is a TensorStaticReason, useful for passing to tensor_static_reason_to_message if needed.
    """
    from .. import config
    from ..source import is_from_unspecialized_param_buffer_source

    if (
        tensor_source.guard_source.is_specialized_nn_module()
        or tensor_source.guard_source.is_unspecialized_builtin_nn_module()
    ) and config.force_nn_module_property_static_shapes:
        return True, TensorStaticReason.NN_MODULE_PROPERTY

    if (
        type(tensor) is torch.nn.Parameter
        or is_from_unspecialized_param_buffer_source(tensor_source)
    ) and config.force_parameter_static_shapes:
        return True, TensorStaticReason.PARAMETER
    if not is_tensor:
        return True, TensorStaticReason.NOT_TENSOR
    return False, None


def lazy_format_graph_tabular(fn_name: str, gm: torch.fx.GraphModule) -> Any:
    def inner() -> str:
        try:
            from tabulate import tabulate  # TODO: Check that this is installed
        except ImportError:
            return (
                "Tabulate module missing, please install tabulate to log the graph in tabular format, logging code instead:\n"
                + str(lazy_format_graph_code(fn_name, gm))
            )

        node_specs = [
            [n.op, n.name, n.target, n.args, n.kwargs] for n in gm.graph.nodes
        ]
        graph_str = tabulate(
            node_specs, headers=["opcode", "name", "target", "args", "kwargs"]
        )
        return _format_graph_code(fn_name, gm.forward.__code__.co_filename, graph_str)

    return LazyString(inner)


def format_bytecode(
    prefix: str, name: str, filename: str, line_no: int, code: Any
) -> str:
    return f"{prefix} {name} {filename} line {line_no} \n{dis.Bytecode(code).dis()}\n"


forward_hook_names = [
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks_with_kwargs",
    "_forward_hooks",
]
backward_hook_names = ["_backward_pre_hooks", "_backward_hooks"]
state_dict_hook_names = [
    "_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
]
all_hook_names = forward_hook_names + backward_hook_names + state_dict_hook_names


def nn_module_has_global_hooks() -> bool:
    # This is limited to backward hooks for now because NNModuleVariable
    # supports fwd hooks underneath.
    return bool(
        len(torch.nn.modules.module._global_backward_hooks)
        or len(torch.nn.modules.module._global_backward_pre_hooks)
    )


def nn_module_get_all_hooks(
    mod: torch.nn.Module,
    check_forward_hooks: bool = False,
    check_backward_hooks: bool = False,
    check_state_dict_hooks: bool = False,
) -> list[Any]:
    """
    Sometimes its useful to differentiate between types of hooks such as forward/backward/pre
    hooks executed during module.__call__, and state_dict hooks which are executed separately.
    """
    hook_dicts_to_check = []
    check_all_hooks = (
        not check_forward_hooks
        and not check_backward_hooks
        and not check_state_dict_hooks
    )
    if check_forward_hooks or check_all_hooks:
        hook_dicts_to_check.extend(forward_hook_names)
    if check_backward_hooks or check_all_hooks:
        hook_dicts_to_check.extend(backward_hook_names)
    if check_state_dict_hooks:
        hook_dicts_to_check.extend(state_dict_hook_names)

    all_hooks = []
    for hook_dict_name in hook_dicts_to_check:
        hooks = getattr(mod, hook_dict_name, [])
        for hook_name in hooks:
            hook = hooks[hook_name]

            all_hooks.append(hook)
    return all_hooks


def nnmodule_has_hooks(
    mod: torch.nn.Module,
    check_forward_hooks: bool = False,
    check_backward_hooks: bool = False,
    check_state_dict_hooks: bool = False,
) -> bool:
    """
    Helper function to check if a module has any hooks attached to it.
    """
    hooks = nn_module_get_all_hooks(
        mod,
        check_forward_hooks=check_forward_hooks,
        check_backward_hooks=check_backward_hooks,
        check_state_dict_hooks=check_state_dict_hooks,
    )
    return bool(hooks)


def invalid_removeable_handle() -> RemovableHandle:
    # need a subclass so weakref works
    class Invalid(dict):  # type: ignore[type-arg]
        pass

    return RemovableHandle(Invalid())


def nn_module_proxy(mod: Any) -> Any:
    if not isinstance(mod, torch.nn.Module):
        return mod
    if isinstance(mod, torch.fx.GraphModule):
        # Dynamo-generated GM's shouldn't contain user-created GM's
        return mod
    proxy = mod.__class__.__new__(mod.__class__)
    proxy.__dict__ = mod.__dict__
    return proxy


class GmWrapper(torch.nn.Module):
    def __init__(
        self, gm: torch.fx.GraphModule, unflatten_fn: Callable[[list[Any]], Any]
    ) -> None:
        super().__init__()
        self.gm = gm
        self.unflatten_fn = unflatten_fn

    def forward(self, *args: Any) -> Any:
        # pyrefly: ignore [redefinition]
        args: list[Any] = list(args)
        return self.gm(*self.unflatten_fn(args))


def flatten_graph_inputs(
    gm: torch.fx.GraphModule, inputs: Any, compile_gm: Callable[[Any, Any], Any]
) -> Callable[..., Any]:
    """
    Mutate inputs so that they are flat and wrap gm such that it
    accepts those inputs.  This is needed for graphs that take
    bumpy inputs.
    """
    inputs_idx_to_clear = [
        i
        for i, node in enumerate(gm.graph.nodes)
        if node.op == "placeholder" and node.meta.get("steal_arg", False)
    ]

    if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
        # fast path, avoid pytree overhead
        # compiled autograd inputs are always a list of tensors, maybe followed by symints
        assert inputs_idx_to_clear == [0]
        assert isinstance(inputs[0], list)
        boxed_inputs_count = len(inputs[0])

        def flatten_fn(args: Any) -> Any:
            return args[0] + list(args[1:])

        def unflatten_fn(flat_args: Any) -> Any:
            return (flat_args[:boxed_inputs_count], *flat_args[boxed_inputs_count:])

        compiled_fn = compile_gm(GmWrapper(gm, unflatten_fn), flatten_fn(inputs))
    else:
        # slow path, don't know inputs structure
        flat_inputs, spec = pytree.tree_flatten(inputs)
        unflatten_fn = functools.partial(pytree.tree_unflatten, treespec=spec)
        compiled_fn = compile_gm(GmWrapper(gm, unflatten_fn), flat_inputs)
        # note this doesn't check the spec, assuming it is the same
        flatten_fn = pytree.arg_tree_leaves

    def wrapper(*args: Any) -> Any:
        flat_args = flatten_fn(args)

        # flat_args is a new list, so we need to clear references from the old list
        for i in inputs_idx_to_clear:
            args[i].clear()

        # this call is boxed to avoid increasing refcount until we reach aot_module_simplified forward
        return compiled_fn(flat_args)

    return wrapper


def get_locals_to_steal(maybe_gm: Any) -> list[Any]:
    if not isinstance(maybe_gm, torch.fx.GraphModule) or not hasattr(maybe_gm, "meta"):
        return []
    return maybe_gm.meta.get("locals_to_steal", [])


def set_locals_to_steal(gm: torch.fx.GraphModule, locals_to_steal: list[Any]) -> None:
    gm.meta["locals_to_steal"] = locals_to_steal


def does_not_override_dict_iter_methods(user_cls: Any) -> bool:
    return (
        user_cls.items in (dict.items, OrderedDict.items)
        and user_cls.values in (dict.values, OrderedDict.values)
        and user_cls.keys in (dict.keys, OrderedDict.keys)
        and user_cls.__iter__ in (dict.__iter__, OrderedDict.__iter__)
    )
