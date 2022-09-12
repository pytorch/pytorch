import enum

__all__ = [
    "QuantType",
    "quant_type_to_str",
]

# Quantization type (dynamic quantization, static quantization).
# Should match the c++ enum in quantization_type.h
class QuantType(enum.IntEnum):
    DYNAMIC = 0
    STATIC = 1
    QAT = 2
    WEIGHT_ONLY = 3

_quant_type_to_str = {
    QuantType.STATIC: "static",
    QuantType.DYNAMIC: "dynamic",
    QuantType.QAT: "qat",
    QuantType.WEIGHT_ONLY: "weight_only",
}

# TODO: make this private
def quant_type_to_str(quant_type: QuantType) -> str:
    return _quant_type_to_str[quant_type]

def _quant_type_from_str(name: str) -> QuantType:
    for quant_type, s in _quant_type_to_str.items():
        if name == s:
            return quant_type
    raise ValueError("Unknown QuantType name '%s'" % name)
