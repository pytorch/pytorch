# flake8: noqa
import torch


# binary ops: <<, >>, |, &, ~, ^

a = torch.ones(3, dtype=torch.float64)
i = int()

i | a  # E: Unsupported operand types
