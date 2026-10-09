from typing_extensions import assert_type

import torch
import torch.nn.functional as F
from torch import Tensor


x = torch.randn(2, 3, 4)

assert_type(F.normalize(x), Tensor)
assert_type(F.normalize(x, dim=(0, 1)), Tensor)
assert_type(F.normalize(x, dim=[-2, -1]), Tensor)
