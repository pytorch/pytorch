import json
import re
from typing import Any, Callable, Dict, List, Union

import attr

from torch.onnx._internal.diagnostics.infra import sarif_om

# A list of types in the SARIF OM to support pretty printing.
# This is solely for type annotation for the functions below.
_SarifClass = Union[
    sarif_om.SarifLog,
    sarif_om.Run,
    sarif_om.ReportingDescriptor,
    sarif_om.Result,
]


def _camel_case_to_snake_case(s: str) -> str:
    return re.sub(r"([A-Z])", r"_\1", s).lower()


def _kebab_case_to_snake_case(s: str) -> str:
    return s.replace("-", "_")


def _convert_key(
    object: Union[Dict[str, Any], Any], convert: Callable[[str], str]
) -> Union[Dict[str, Any], Any]:
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
        new_dict[new_k] = new_v
    return new_dict


def sarif_to_json(attr_cls_obj: _SarifClass) -> str:
    dict = attr.asdict(attr_cls_obj)
    dict = _convert_key(dict, _camel_case_to_snake_case)
    return json.dumps(dict, indent=4)
