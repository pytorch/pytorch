import torch
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import APoTQuantizer
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

class APoTFakeQuantize(FakeQuantizeBase):
    observer: APoTObserver

    def __init__(self, observer: APoTObserver):
        super().__init__()
        self.observer = observer

    def calculate_qparams(self, signed: bool, min_val=None, max_val=None):
        qparams = self.observer.calculate_qparams(signed=signed, min_val=min_val, max_val=max_val)

        return qparams

    def forward(self, X: torch.Tensor, signed: bool):
        if self.observer_enabled[0] == 1:
            min_val, max_val = torch.aminmax(X)
            qparams = self.calculate_qparams(signed, min_val, max_val)
        if self.fake_quant_enabled[0] == 1:
            quantizer = APoTQuantizer(self.observer.b, self.observer.k, self.observer.min_val, self.observer.max_val, signed)

        X = quantizer.quantize_APoT(X)
        X = quantizer.dequantize(X)

        return X
