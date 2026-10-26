from __future__ import annotations

import torch


class FutureAnnotatedBase(torch.nn.Module):
    inherited: torch.Tensor
