import torch
import tensorflow as tf
from torch import Tensor
from torch.ao.quantization.experimental.observer import APoTObserver

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

        apot_result = torch.Tensor([])

        num_rows, num_columns = tensor2quantize.get_shape()

        # traverse tensor2quantize, quantize each element
        for x in tensor2quantize:
            x_apot = obs.float_to_apot(x, quantized_levels, level_indices)
            apot_result = torch.cat(apot_result, x_apot)

        torch.reshape(apot_result, (num_rows, num_columns))

        print(apot_result)


    def dequantize(self) -> Tensor:
        raise NotImplementedError

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
