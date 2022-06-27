import torch
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import APoTQuantizer, quantize_APoT, dequantize_APoT
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

class APoTFakeQuantize(FakeQuantizeBase):
    observer: APoTObserver

    def __init__(self, observer: APoTObserver):
        super().__init__()
        self.observer = observer

    def calculate_qparams(self, signed: bool, min_val=None, max_val=None):  # type: ignore[override]
        return self.observer.calculate_qparams(signed=signed, min_val=min_val, max_val=max_val)

    def forward(self, X: torch.Tensor, signed: bool):  # type: ignore[override]
        alpha = gamma = quantization_levels = level_indices = None

        if self.observer_enabled[0] == 1:
            self.observer.forward(X)
            alpha, gamma, quantization_levels, level_indices = self.observer.calculate_qparams(signed)
        if self.fake_quant_enabled[0] == 1:
            assert (alpha is not None
                    and gamma is not None
                    and quantization_levels is not None
                    and level_indices is not None), "Must set qparams for fake quant"

            quantizer = APoTQuantizer(alpha, gamma, quantization_levels, level_indices)
            X = quantize_APoT(X, alpha, gamma, quantization_levels, level_indices)
            X = dequantize_APoT(X, alpha, gamma, quantization_levels, level_indices)

        return X
