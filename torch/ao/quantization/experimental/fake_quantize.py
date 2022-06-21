import torch
from torch.ao.quantization import FakeQuantizeBase
from torch.ao.quantization.experimental.observer import APoTObserver

class APoTFakeQuantize(FakeQuantizeBase):
    max_val: float
    b: int
    k: int
    n: int
    alpha: float
    gamma: float
    level_indices: torch.Tensor
    quantization_levels: torch.Tensor
    observer: APoTObserver

    def __init__(self, observer=APoTObserver, max_val: float, b: int, k: int):
        super().__init__()
        self.b = b
        self.k = k

        # check for valid inputs of b, k
        assert(self.k and self.k != 0)
        assert(self.b % self.k == 0)

        # compute n and store as member variable
        self.n = self.b // self.k

        self.alpha = max_val

        self.observer = observer

    def calculate_qparams(self, signed: bool):
        self.gamma = self.observer.calculate_qparams(signed=signed)[0]
        return self.observer.calculate_qparams(signed=signed)

    def forward():
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X, self.scale, self.zero_point,
                    self.ch_axis, self.quant_min, self.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.quant_min, self.quant_max)
        return X
