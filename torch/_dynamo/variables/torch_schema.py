import functools
import inspect
from collections.abc import Callable
from typing import Any

import torch


MutatedArgInfo = tuple[int, str, bool, bool]
InplaceParamInfo = tuple[str, str, int]


def _is_tensor_list_type(typ: Any) -> bool:
    return isinstance(typ, torch.ListType) and isinstance(
        typ.getElementType(), torch.TensorType
    )


@functools.lru_cache(None)
def torch_op_mutates_first_arg(name: str) -> bool:
    if "." in name:
        name = name.split(".", 1)[0]
    if not name.startswith("aten::"):
        name = f"aten::{name}"
    return any(
        schema.arguments
        and isinstance(schema.arguments[0].type, torch.TensorType)
        and schema.arguments[0].alias_info
        and schema.arguments[0].alias_info.is_write
        for schema in torch._C._jit_get_schemas_for_operator(name)
    )


def _schema_mutated_arg_infos(schema: Any) -> tuple[MutatedArgInfo, ...]:
    return tuple(
        (
            idx,
            arg.name,
            _is_tensor_list_type(arg.type),
            arg.kwarg_only,
        )
        for idx, arg in enumerate(schema.arguments)
        if arg.alias_info
        and arg.alias_info.is_write
        and (isinstance(arg.type, torch.TensorType) or _is_tensor_list_type(arg.type))
    )


@functools.lru_cache(None)
def torch_function_mutated_arg_infos(
    fn: Callable[..., Any],
) -> tuple[MutatedArgInfo, ...]:
    schema = getattr(fn, "_schema", None)
    if schema is not None:
        return _schema_mutated_arg_infos(schema)

    schemas = getattr(fn, "_schemas", None)
    if schemas is not None:
        return tuple(
            info
            for schema in schemas.values()
            for info in _schema_mutated_arg_infos(schema)
        )

    if getattr(fn, "__module__", None) not in ("torch", "torch._C._nn"):
        return ()

    fn_name = getattr(fn, "__name__", None)
    if not fn_name:
        return ()

    infos = {
        info
        for schema in torch._C._jit_get_schemas_for_operator(f"aten::{fn_name}")
        for info in _schema_mutated_arg_infos(schema)
    }
    # Public torch functions use `input=` for the first tensor argument even
    # when the underlying ATen schema calls it `self`.
    infos.update(
        (idx, "input", is_tensor_list, kwarg_only)
        for idx, name, is_tensor_list, kwarg_only in tuple(infos)
        if idx == 0 and name == "self"
    )
    return tuple(infos)


@functools.lru_cache(None)
def torch_function_inplace_param_names(
    fn: Callable[..., Any],
) -> InplaceParamInfo | None:
    if getattr(fn, "__module__", None) != "torch.nn.functional":
        return None

    try:
        params = tuple(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None
    if not params or "inplace" not in params:
        return None
    return params[0], "inplace", params.index("inplace")
