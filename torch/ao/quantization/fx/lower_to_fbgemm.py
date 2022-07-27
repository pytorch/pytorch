from ._lower_to_native_backend import _lower_to_native_backend
from .graph_module import QuantizedGraphModule
from ..qconfig import QConfigAny
from typing import Dict, Tuple

__all__ = ['lower_to_fbgemm']

def lower_to_fbgemm(
    model: QuantizedGraphModule,
    qconfig_map: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str, type]]
) -> QuantizedGraphModule:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to fbgemm
    """
    return _lower_to_native_backend(model, qconfig_map, node_name_to_scope)
