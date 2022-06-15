import torch
from torch import Tensor
from torch.ao.quantization.experimental.observer import APoTObserver, apot_to_float

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    @staticmethod
    def quantize_APoT(tensor2quantize: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def dequantize(tensor2dequantize: Tensor, b: int, k: int) -> Tensor:  # type: ignore[override]
        tensor2dequantize = tensor2dequantize.float()

        max_val = torch.max(tensor2dequantize)

        # make observer
        obs = APoTObserver(max_val=max_val, b=b, k=k)
        obs_result = obs.calculate_qparams(signed=False)

        quantized_levels = obs_result[1]
        level_indices = obs_result[2]

        # map apot_to_float over tensor2quantize elements
        result = tensor2dequantize.apply_(lambda x: float(apot_to_float(x, quantized_levels, level_indices)))

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
