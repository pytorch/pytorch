import torch
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

class APoTFakeQuantize():
    max_val: float
    b: int
    k: int
    n: int
    gamma: float
    level_indices: torch.Tensor
    quantization_levels: torch.Tensor
    observer: APoTObserver

    def __init__(self, observer: APoTObserver):
        self.observer = observer
        self.b = observer.b
        self.k = observer.k
        self.max_val = observer.max_val

    def calculate_qparams(self, signed: bool):
        qparams = self.observer.calculate_qparams(signed=signed)
        self.gamma = qparams[0]
        self.quantization_levels = qparams[1]
        self.level_indices = qparams[2]

        return qparams

    def forward(self, X: torch.Tensor, signed: bool):
        quantizer = APoTQuantizer(self.b, self.k, self.max_val, signed)

        X = quantizer.quantize_APoT(X)
        X = quantizer.dequantize(X)

        return X
