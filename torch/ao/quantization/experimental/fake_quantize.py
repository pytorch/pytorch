import torch
from torch.ao.quantization import FakeQuantizeBase
from torch.ao.quantization.experimental.observer import APoTObserver

class APoTFakeQuantize(FakeQuantizeBase):
    def __init__(self, observer=APoTObserver):
        super().__init__()

    def calculate_qparams(self):
        return APoTObserver.calculate_qparams

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
