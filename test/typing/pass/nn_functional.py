# flake8: noqa
import torch


torch.nn.functional.mse_loss(
    torch.ones(1), torch.ones(1), weight=torch.ones(1)
)
from typing_extensions import assert_type

import torch
import torch.nn.functional as F
from torch import Tensor


x = torch.randn(2, 3, 4)

assert_type(F.normalize(x), Tensor)
assert_type(F.normalize(x, dim=(0, 1)), Tensor)
assert_type(F.normalize(x, dim=[-2, -1]), Tensor)
