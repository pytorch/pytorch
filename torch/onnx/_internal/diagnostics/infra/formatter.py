from __future__ import annotations

import dataclasses
import json
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics.infra import sarif


# A list of types in the SARIF module to support pretty printing.
# This is solely for type annotation for the functions below.
_SarifClass = Union[
    sarif.SarifLog,
    sarif.Run,
    sarif.ReportingDescriptor,
    sarif.Result,
]


class LazyString:
    """A class to lazily evaluate a string.

    Adopted from `torch._dynamo.utils.LazyString`.
    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.func(*self.args, **self.kwargs)


def lazy_format_exception(exception: Exception) -> LazyString:
    return LazyString(
        lambda: "\n".join(
            (
                "```",
                *traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                ),
                "```",
            )
        ),
    )


@_beartype.beartype
def snake_case_to_camel_case(s: str) -> str:
    splits = s.split("_")
    if len(splits) <= 1:
        return s
    return "".join([splits[0], *map(str.capitalize, splits[1:])])


@_beartype.beartype
def camel_case_to_snake_case(s: str) -> str:
    return re.sub(r"([A-Z])", r"_\1", s).lower()


@_beartype.beartype
def kebab_case_to_snake_case(s: str) -> str:
    return s.replace("-", "_")


@_beartype.beartype
def _convert_key(
    object: Union[Dict[str, Any], Any], convert: Callable[[str], str]
) -> Union[Dict[str, Any], Any]:
    """Convert and update keys in a dictionary with "convert".

    Any value that is a dictionary will be recursively updated.
    Any value that is a list will be recursively searched.

    Args:
        object: The object to update.
        convert: The function to convert the keys, e.g. `kebab_case_to_snake_case`.

    Returns:
        The updated object.
    """
    if not isinstance(object, Dict):
        return object
    new_dict = {}
    for k, v in object.items():
        new_k = convert(k)
        if isinstance(v, Dict):
            new_v = _convert_key(v, convert)
        elif isinstance(v, List):
            new_v = [_convert_key(elem, convert) for elem in v]
        else:
            new_v = v
        if new_v is None:
            # Otherwise unnecessarily bloated sarif log with "null"s.
            continue
        if new_v == -1:
            # WAR: -1 as default value shouldn't be logged into sarif.
            continue

        new_dict[new_k] = new_v

    return new_dict


@_beartype.beartype
def sarif_to_json(attr_cls_obj: _SarifClass, indent: Optional[str] = " ") -> str:
    dict = dataclasses.asdict(attr_cls_obj)
    dict = _convert_key(dict, snake_case_to_camel_case)
    return json.dumps(dict, indent=indent, separators=(",", ":"))


@_beartype.beartype
def format_argument(obj: Any) -> str:
    return f"{type(obj)}"


@_beartype.beartype
def display_name(fn: Callable) -> str:
    if hasattr(fn, "__qualname__"):
        return fn.__qualname__
    elif hasattr(fn, "__name__"):
        return fn.__name__
    else:
        return str(fn)
