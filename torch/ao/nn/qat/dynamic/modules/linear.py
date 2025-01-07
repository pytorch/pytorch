from typing import Optional, TYPE_CHECKING, Union

import torch


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import QConfig  # noqa: TC004


__all__ = ["Linear"]


class Linear(torch.ao.nn.qat.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for dynamic quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qconfig: Optional["QConfig"] = None,
        device: Optional[Union[int, str, torch.device]] = None,
        dtype: Optional[str] = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, qconfig, device, dtype)
        if not torch.ao.quantization.qconfig._activation_is_memoryless(qconfig):  # type: ignore[arg-type]
            raise ValueError(
                "Dynamic QAT requires a memoryless observer."
                + "This means a MovingAverage observer with averaging constant equal to 1"
            )
