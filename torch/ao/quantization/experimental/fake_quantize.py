import torch
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

class APoTFakeQuantize():
    max_val: float
    b: int
    k: int
    n: int
    alpha: float
    gamma: float
    signed: bool
    level_indices: torch.Tensor
    quantization_levels: torch.Tensor
    observer: APoTObserver

    def __init__(self, observer: APoTObserver):
        super().__init__()

        self.observer = observer
        self.b = observer.b
        self.k = observer.k
        self.signed = observer.signed
        self.max_val = observer.max_val

    def calculate_qparams(self, signed: bool):
        self.observer = self.observer.calculate_qparams(signed=signed)[0]
        self.gamma = self.observer[0]
        self.quantization_levels = self.observer[1]
        self.level_indices = self.observer[2]

        return self.observer

    def forward(self, X):
        quantizer = APoTQuantizer(self.observer)

        X = quantizer.quantize_APoT(X)
        X = quantizer.dequantize(X)

        return X
