import torch
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import APoTQuantizer
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

class APoTFakeQuantize(FakeQuantizeBase):
    observer: APoTObserver

    def __init__(self, observer: APoTObserver):
        super().__init__()
        self.observer = observer

    def calculate_qparams(self, signed: bool):
        qparams = self.observer.calculate_qparams(signed=signed)
        self.gamma = qparams[0]
        self.quantization_levels = qparams[1]
        self.level_indices = qparams[2]

        return qparams

    def forward(self, X: torch.Tensor, signed: bool):
        quantizer = APoTQuantizer(self.observer.b, self.observer.k, self.observer.max_val, signed)

        X = quantizer.quantize_APoT(X)
        X = quantizer.dequantize(X)

        return X
