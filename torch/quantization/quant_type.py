import enum

# Quantization type (dynamic quantization, static quantization).
# Should match the c++ enum in quantization_type.h
class QuantType(enum.IntEnum):
    DYNAMIC = 0
    STATIC = 1
    QAT = 2
    WEIGHT_ONLY = 3
    ACTIVATION_ONLY = 4


def quant_type_to_str(quant_type):
    """Converts the QuantType to its string equivalent.

    Note: The default is different to its C++ equivalent. Here we return None,
          while in C++ unknown key would set the error state flag.
    """
    m = {
        QuantType.STATIC: "static",
        QuantType.DYNAMIC: "dynamic",
        QuantType.QAT: "qat",
        QuantType.WEIGHT_ONLY: "weight_only",
        QuantType.ACTIVATION_ONLY: "activation_only",
    }
    return m.get(quant_type, None)
