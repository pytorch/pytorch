from __future__ import annotations

import dataclasses
import json
import re
import traceback
from typing import Any, Callable, Union

from torch._logging import LazyString
from torch.onnx._internal.diagnostics.infra import sarif


# A list of types in the SARIF module to support pretty printing.
# This is solely for type annotation for the functions below.
_SarifClass = Union[
    sarif.SarifLog,
    sarif.Run,
    sarif.ReportingDescriptor,
    sarif.Result,
]


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


def snake_case_to_camel_case(s: str) -> str:
    splits = s.split("_")
    if len(splits) <= 1:
        return s
    return "".join([splits[0], *map(str.capitalize, splits[1:])])


def camel_case_to_snake_case(s: str) -> str:
    return re.sub(r"([A-Z])", r"_\1", s).lower()


def kebab_case_to_snake_case(s: str) -> str:
    return s.replace("-", "_")


def _convert_key(
    object: dict[str, Any] | Any, convert: Callable[[str], str]
) -> dict[str, Any] | Any:
    """Convert and update keys in a dictionary with "convert".

    Any value that is a dictionary will be recursively updated.
    Any value that is a list will be recursively searched.

    Args:
        object: The object to update.
        convert: The function to convert the keys, e.g. `kebab_case_to_snake_case`.

    Returns:
        The updated object.
    """
    if not isinstance(object, dict):
        return object
    new_dict = {}
    for k, v in object.items():
        new_k = convert(k)
        if isinstance(v, dict):
            new_v = _convert_key(v, convert)
        elif isinstance(v, list):
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


def sarif_to_json(attr_cls_obj: _SarifClass, indent: str | None = " ") -> str:
    dict = dataclasses.asdict(attr_cls_obj)
    dict = _convert_key(dict, snake_case_to_camel_case)
    return json.dumps(dict, indent=indent, separators=(",", ":"))


def format_argument(obj: Any) -> str:
    return f"{type(obj)}"


def display_name(fn: Callable) -> str:
    if hasattr(fn, "__qualname__"):
        return fn.__qualname__
    elif hasattr(fn, "__name__"):
        return fn.__name__
    else:
        return str(fn)
