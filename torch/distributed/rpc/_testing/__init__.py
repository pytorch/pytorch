from __future__ import absolute_import, division, print_function, unicode_literals

import torch


def is_available():
    return hasattr(torch._C, "_faulty_agent_init")


if is_available() and not torch._C._faulty_agent_init():
    raise RuntimeError("Failed to initialize torch.distributed.rpc._testing")

if is_available():
    from . import faulty_agent_backend_registry
