import os
import os.path
import typing

import torch


if hasattr(typing, "io"):
    import typing.io
    import typing.re


class Module(torch.nn.Module):
    def forward(self):
        return os.path.abspath("test")
