import torch
import tensorflow as tf
from torch import Tensor
from torch.ao.quantization.experimental.observer import APoTObserver, float_to_apot

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    @staticmethod
    def quantize_APoT(tensor2quantize: Tensor, b: int, k: int) -> Tensor:
        max_val = torch.max(tensor2quantize)

        # make observer
        obs = APoTObserver(max_val=max_val, b=b, k=k)
        obs_result = obs.calculate_qparams(signed=False)

        quantized_levels = obs_result[1]
        level_indices = obs_result[2]

        # map apot_to_float over tensor2quantize elements
        result = tf.map_fn(fn=lambda t: float_to_apot(t, quantized_levels, level_indices), elems=tensor2quantize, dtype=tf.float32)

        return result

    def dequantize(self) -> Tensor:
        raise NotImplementedError

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
