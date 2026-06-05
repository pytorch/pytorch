import functools
import inspect
from collections.abc import Callable
from typing import Any

import torch


MutatedFirstArgInfo = tuple[str, bool]
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


def _schema_mutated_first_arg_info(schema: Any) -> MutatedFirstArgInfo | None:
    if (
        schema.arguments
        and schema.arguments[0].alias_info
        and schema.arguments[0].alias_info.is_write
        and (
            isinstance(schema.arguments[0].type, torch.TensorType)
            or _is_tensor_list_type(schema.arguments[0].type)
        )
    ):
        return (
            schema.arguments[0].name,
            _is_tensor_list_type(schema.arguments[0].type),
        )
    return None


@functools.lru_cache(None)
def torch_function_mutated_first_arg_infos(
    fn: Callable[..., Any],
) -> tuple[MutatedFirstArgInfo, ...]:
    schema = getattr(fn, "_schema", None)
    if schema is not None:
        info = _schema_mutated_first_arg_info(schema)
        return (info,) if info else ()

    schemas = getattr(fn, "_schemas", None)
    if schemas is not None:
        return tuple(
            info
            for schema in schemas.values()
            if (info := _schema_mutated_first_arg_info(schema))
        )

    if getattr(fn, "__module__", None) not in ("torch", "torch._C._nn"):
        return ()

    fn_name = getattr(fn, "__name__", None)
    if not fn_name:
        return ()

    infos = {
        info
        for schema in torch._C._jit_get_schemas_for_operator(f"aten::{fn_name}")
        if (info := _schema_mutated_first_arg_info(schema))
    }
    # Public torch functions use `input=` for the first tensor argument even
    # when the underlying ATen schema calls it `self`.
    infos.update(("input", is_tensor_list) for _, is_tensor_list in tuple(infos))
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
