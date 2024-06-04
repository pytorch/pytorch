# mypy: allow-untyped-defs
import torch
from torch import Tensor
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.ao.quantization.experimental.fake_quantize_function import fake_quantize_function

class APoTFakeQuantize(FakeQuantizeBase):
    alpha: Tensor
    gamma: Tensor
    quantization_levels: Tensor
    level_indices: Tensor

    def __init__(self, observer=APoTObserver, **observer_kwargs):
        super().__init__()
        self.activation_post_process = observer(**observer_kwargs)
        self.dtype = self.activation_post_process.dtype

    def calculate_qparams(self, signed=False):  # type: ignore[override]
        return self.activation_post_process.calculate_qparams(signed=signed)

    def forward(self, X: torch.Tensor):  # type: ignore[override]
        if self.observer_enabled[0] == 1:
            self.activation_post_process.forward(X)
            result = self.activation_post_process.calculate_qparams(signed=False)
            self.alpha = result[0]
            self.gamma = result[1]
            self.quantization_levels = result[2]
            self.level_indices = result[3]

        if self.fake_quant_enabled[0] == 1:
            assert (self.alpha is not None
                    and self.gamma is not None
                    and self.quantization_levels is not None
                    and self.level_indices is not None), "Must set qparams for fake quant"

            X = fake_quantize_function.apply(X, self.alpha, self.gamma, self.quantization_levels, self.level_indices)

        return X
