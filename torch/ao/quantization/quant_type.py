import enum

# Quantization type (dynamic quantization, static quantization).
# Should match the c++ enum in quantization_type.h
class QuantType(enum.IntEnum):
    DYNAMIC = 0
    STATIC = 1
    QAT = 2
    WEIGHT_ONLY = 3


def quant_type_to_str(quant_type):
    m = {
        QuantType.STATIC: "static",
        QuantType.DYNAMIC: "dynamic",
        QuantType.QAT: "qat",
        QuantType.WEIGHT_ONLY: "weight_only",
    }
    return m[quant_type]
