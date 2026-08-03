# mypy: ignore-errors

import torch.nn as nn


class Net(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 20)
