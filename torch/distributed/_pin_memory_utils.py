# mypy: allow-untyped-defs
import logging
import os
from logging import getLogger

import torch

logger = getLogger()
logger.setLevel(logging.INFO)


# Allows retrying cudaHostRegister if it fails
CKPT_PIN_ALLOW_RETRY = os.environ.get("CKPT_PIN_ALLOW_RETRY", "1") == "1"
# Peeks last cudaError before pinning shared memory
CKPT_PIN_PEEK_CUDA_ERROR = os.environ.get("CKPT_PIN_PEEK_CUDA_ERROR", "0") == "1"
# Pops last cudaError before pinning shared memory
CKPT_PIN_POP_CUDA_ERROR = os.environ.get("CKPT_PIN_POP_CUDA_ERROR", "0") == "1"


def pin_shared_mem(data_ptr: int, size: int) -> None:
    cudart = torch.cuda.cudart()

    # It's not ideal to do this much error handling, we should consider removing this code once things are a bit more stable.
    if CKPT_PIN_PEEK_CUDA_ERROR:
        if hasattr(cudart, "cudaPeekAtLastError"):
            err = cudart.cudaPeekAtLastError()
            if err != 0:
                logger.warning(
                    f"Cuda cudaPeekAtLastError returned non-zero error code: {err} when trying to pin shared memory!\n"
                    "There is likely a previous error that has been ignored, but this error will likely cause a failure later on."
                    "To disable this warning, set 'CKPT_PIN_PEEK_CUDA_ERROR' to 0"
                )
        else:
            logger.info(
                f"Ignoring {CKPT_PIN_POP_CUDA_ERROR=}, since it's not available on this version of torch. Please update."
            )

    if CKPT_PIN_POP_CUDA_ERROR:
        if hasattr(cudart, "cudaGetLastError"):
            err = cudart.cudaGetLastError()
            if err != 0:
                logger.warning(
                    f"Cuda cudaGetLastError returned non-zero error code: {err} when trying to pin shared memory!\n"
                    "There is likely a previous error that has been ignored, but this error will likely cause a failure later on."
                    f"Clearing the error for now since `{CKPT_PIN_POP_CUDA_ERROR=}`"
                )
        else:
            logger.info(
                f"Ignoring {CKPT_PIN_POP_CUDA_ERROR=}, since it's not available on this version of torch. Please update."
            )

    succ = int(
        cudart.cudaHostRegister(
            data_ptr,
            size,
            1,  # lines up with 'cudaHostRegisterPortable'
        )
    )

    if succ != 0:
        raise RuntimeError(f"Registering shared memory failed with cudaError: {succ}\n")


def unpin_memory(data_ptr) -> None:
    succ = int(torch.cuda.cudart().cudaHostUnregister(data_ptr))
    assert succ == 0, f"Unpinning shared memory failed with error-code: {succ}"

