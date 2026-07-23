import os
import os.path
import typing
import typing.io
import typing.re  # noqa: F401

import torch


class Module(torch.nn.Module):
    def forward(self):
        return os.path.abspath("test")
