# mypy: allow-untyped-defs
import torch


class QuantizedLinear(torch.jit.ScriptModule):
    def __init__(self, other):
        raise RuntimeError(
            "torch.jit.QuantizedLinear is no longer supported. Please use "
            "torch.ao.nn.quantized.dynamic.Linear instead."
        )


# FP16 weights
class QuantizedLinearFP16(torch.jit.ScriptModule):
    def __init__(self, other):
        super().__init__()
        raise RuntimeError(
            "torch.jit.QuantizedLinearFP16 is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.Linear instead."
        )


# Quantized RNN cell implementations
class QuantizedRNNCellBase(torch.jit.ScriptModule):
    def __init__(self, other):
        raise RuntimeError(
            "torch.jit.QuantizedRNNCellBase is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.RNNCell instead."
        )


class QuantizedRNNCell(QuantizedRNNCellBase):
    def __init__(self, other):
        raise RuntimeError(
            "torch.jit.QuantizedRNNCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.RNNCell instead."
        )


class QuantizedLSTMCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super().__init__(other)
        raise RuntimeError(
            "torch.jit.QuantizedLSTMCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.LSTMCell instead."
        )


class QuantizedGRUCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super().__init__(other)
        raise RuntimeError(
            "torch.jit.QuantizedGRUCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.GRUCell instead."
        )


class QuantizedRNNBase(torch.jit.ScriptModule):
    def __init__(self, other, dtype=torch.int8):
        raise RuntimeError(
            "torch.jit.QuantizedRNNBase is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic instead."
        )


class QuantizedLSTM(QuantizedRNNBase):
    def __init__(self, other, dtype):
        raise RuntimeError(
            "torch.jit.QuantizedLSTM is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.LSTM instead."
        )


class QuantizedGRU(QuantizedRNNBase):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "torch.jit.QuantizedGRU is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.GRU instead."
        )


def quantize_rnn_cell_modules(module):
    raise RuntimeError(
        "quantize_rnn_cell_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )


def quantize_linear_modules(module, dtype=torch.int8):
    raise RuntimeError(
        "quantize_linear_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )


def quantize_rnn_modules(module, dtype=torch.int8):
    raise RuntimeError(
        "quantize_rnn_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )
