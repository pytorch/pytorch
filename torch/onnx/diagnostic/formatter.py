import json
import re
from typing import Any, Callable, Dict, List, Union

import attr


def sarif_om_to_json(attr_cls_obj) -> str:
    dict = attr.asdict(attr_cls_obj)

    def convert_key(
        object: Union[Dict[str, Any], Any], convert: Callable[[str], str]
    ) -> Union[Dict[str, Any], Any]:
        if not isinstance(object, Dict):
            return object
        new_dict = {}
        for k, v in object.items():
            new_k = convert(k)
            if isinstance(v, Dict):
                new_v = convert_key(v, convert)
            elif isinstance(v, List):
                new_v = [convert_key(elem, convert) for elem in v]
            else:
                new_v = v
            new_dict[new_k] = new_v
        return new_dict

    def from_camel_case_to_snake_case(s: str) -> str:
        return re.sub(r"([A-Z])", r"_\1", s).lower()

    dict = convert_key(dict, from_camel_case_to_snake_case)
    return json.dumps(dict, indent=4)
