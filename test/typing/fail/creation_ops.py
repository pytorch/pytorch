# flake8: noqa
import torch

torch.tensor([3], dtype='int32')  # E: expected "Optional[dtype]"
torch.ones(3, dtype='int32')  # E: No overload variant of "ones" matches argument types "int", "str"
torch.zeros(3, dtype='int32')  # E: No overload variant of "zeros" matches argument types "int", "str"
