# flake8: noqa
import torch


torch.nn.functional.mse_loss(
    torch.ones(1), torch.ones(1), weight=torch.ones(1)
)
