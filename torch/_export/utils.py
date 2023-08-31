import dataclasses

from typing import Any, List, Optional, Tuple

from torch.utils._pytree import (
    _register_pytree_node,
    Context,
    FlattenFunc,
    MaybeFromStrFunc,
    ToStrFunc,
    UnflattenFunc,
)


def register_dataclass_as_pytree_node(
    typ: Any,
    flatten_fn: Optional[FlattenFunc] = None,
    unflatten_fn: Optional[UnflattenFunc] = None,
    to_str_fn: Optional[ToStrFunc] = None,
    maybe_from_str_fn: Optional[MaybeFromStrFunc] = None,
    *,
    return_none_fields: bool = False,
) -> None:
    assert dataclasses.is_dataclass(
        typ
    ), f"Only dataclasses can be registered with this function: {typ}"

    def default_flatten_fn(obj: Any) -> Tuple[List[Any], Context]:
        flattened = []
        flat_names = []
        none_names = []
        for f in dataclasses.fields(obj):
            name, val = f.name, getattr(obj, f.name)
            if val is not None or return_none_fields:
                flattened.append(val)
                flat_names.append(name)
            else:
                none_names.append(name)
        return flattened, (typ, flat_names, none_names)

    def default_unflatten_fn(values: List[Any], context: Context) -> Any:
        typ, flat_names, none_names = context
        return typ(**dict(zip(flat_names, values)), **{k: None for k in none_names})

    flatten_fn = flatten_fn if flatten_fn is not None else default_flatten_fn
    unflatten_fn = unflatten_fn if unflatten_fn is not None else default_unflatten_fn

    _register_pytree_node(
        typ,
        flatten_fn,
        unflatten_fn,
        None,
        None,
    )
