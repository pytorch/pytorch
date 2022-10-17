
import os
import torch


def is_available():
    return hasattr(torch._C, "_faulty_agent_init")


if os.environ.get("PARSH_AUTORELOAD_CONTEXT") != "1":
    if is_available() and not torch._C._faulty_agent_init():
        raise RuntimeError("Failed to initialize torch.distributed.rpc._testing")

if is_available():
    # Registers FAULTY_TENSORPIPE RPC backend.
    from . import faulty_agent_backend_registry
    from torch._C._distributed_rpc_testing import (
        FaultyTensorPipeRpcBackendOptions,
        FaultyTensorPipeAgent,
    )
