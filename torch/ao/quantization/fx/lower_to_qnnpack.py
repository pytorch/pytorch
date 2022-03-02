from ._lower_to_native_backend import _lower_to_native_backend
from .graph_module import QuantizedGraphModule
from ..qconfig import QConfigAny
from typing import Dict

def lower_to_qnnpack(
    model: QuantizedGraphModule,
    qconfig_map: Dict[str, QConfigAny]
) -> QuantizedGraphModule:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to qnnpack
    """
    return _lower_to_native_backend(model, qconfig_map)
