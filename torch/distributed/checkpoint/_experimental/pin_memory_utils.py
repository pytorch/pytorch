# mypy: allow-untyped-defs
import logging
import os
from logging import getLogger

import torch


logger = getLogger()
logger.setLevel(logging.INFO)


# Allows retrying cudaHostRegister if it fails
CKPT_PIN_ALLOW_RETRY = os.environ.get("CKPT_PIN_ALLOW_RETRY", "1") == "1"


def pin_shared_mem(data_ptr: int, size: int) -> None:
    cudart = torch.cuda.cudart()

    succ = int(
        cudart.cudaHostRegister(
            data_ptr,
            size,
            1,  # lines up with 'cudaHostRegisterPortable'
        )
    )

    if succ != 0:
        raise RuntimeError(
            "Registering shared memory failed with cudaError: %s"
            " It's possible that this is an asynchronous error raised from a previous cuda operation."
            " Consider launching with CUDA_LAUNCH_BLOCKING=1 to debug." % succ
        )


def unpin_memory(data_ptr) -> None:
    succ = int(torch.cuda.cudart().cudaHostUnregister(data_ptr))
    assert succ == 0, "Unpinning shared memory failed with error-code: %s" % succ
