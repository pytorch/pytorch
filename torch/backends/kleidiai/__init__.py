# mypy: allow-untyped-defs
import torch


def is_available():
    r"""Return whether PyTorch is built with KleidiAI support."""
    return torch._C.has_kleidiai
