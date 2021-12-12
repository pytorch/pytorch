from ._lower_to_native_backend import _lower_to_native_backend
from .graph_module import QuantizedGraphModule

def lower_to_fbgemm(model: QuantizedGraphModule) -> QuantizedGraphModule:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to fbgemm
    """
    return _lower_to_native_backend(model)
