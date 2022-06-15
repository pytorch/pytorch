import torch
from torch import Tensor

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    @staticmethod
    def quantize_APoT(tensor2quantize: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def dequantize(tensor2dequantize: Tensor, b: int, k: int) -> Tensor:  # type: ignore[override]
        tensor2dequantize = torch.tensor(tensor2dequantize.numpy())

        max_val = torch.max(tensor2dequantize)

        # make observer
        obs = APoTObserver(max_val=max_val, b=b, k=k)
        obs_result = obs.calculate_qparams(signed=False)

        quantized_levels = obs_result[1]
        level_indices = obs_result[2]

        # map apot_to_float over tensor2quantize elements
        result_tf = tf.map_fn(fn=lambda t: apot_to_float(t, quantized_levels, level_indices),
                           elems=tensor2dequantize, dtype=tf.float32)

        result_tf.mark_used()

        result = torch.tensor(result_tf.numpy())

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
