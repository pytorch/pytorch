from .quantization_patterns import (
    QuantizeHandler,
)
# TODO: remove
class CommonQuantizeHandler(QuantizeHandler):
    """ Common quantized op, first input and first output will be quantized
    """
    pass
