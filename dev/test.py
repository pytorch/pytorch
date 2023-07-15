import torch
import torch.nn.functional as F

N = 10
F.group_norm(
    input=torch.randn(1, N, 1, dtype=torch.float64),
    num_groups=1,
    weight=torch.ones(N, dtype=torch.float32),
)