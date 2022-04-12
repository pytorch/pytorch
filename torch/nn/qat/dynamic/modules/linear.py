import torch
from torch.ao.quantization import activation_is_memoryless


class Linear(torch.nn.qat.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for dynamic quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.
    """

    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, qconfig, device, dtype)
        if not activation_is_memoryless(qconfig):
            raise ValueError(
                "Dynamic QAT requires a memoryless observer." +
                "This means a MovingAverage observer with averaging constant equal to 1"
            )
