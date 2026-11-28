from typing_extensions import assert_type

import torch


assert_type(
    torch.empty(1).requires_grad_(requires_grad=False),
    torch.Tensor,
)
