from typing import Any, Final

iso_c_binding_map: Final[dict[str, dict[str, str]]] = ...

isoc_c2pycode_map: Final[dict[str, Any]] = {}  # not implemented
iso_c2py_map: Final[dict[str, Any]] = {}  # not implemented

isoc_kindmap: Final[dict[str, str]] = ...

# namespace pollution
c_type: str
c_type_dict: dict[str, str]
fortran_type: str
