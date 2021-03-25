# flake8: noqa
import torch

a = torch.ones(3, dtype=torch.float64)
i = int()

# shift left (<<)

# shift right (>>)


# binary or (|)
i | a  # E: Unsupported operand types

# binary and (&)

# complement (~)

# xor (^)