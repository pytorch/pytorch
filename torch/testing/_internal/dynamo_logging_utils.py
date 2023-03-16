import torch._logging._registrations
import functools
from . import logging_utils

make_logging_test = functools.partial(
    logging_utils.make_test,
    log_names=(
        torch._logging._registrations.TORCHDYNAMO_LOG_NAME,
        torch._logging._registrations.TORCHINDUCTOR_LOG_NAME,
        torch._logging._registrations.AOT_AUTOGRAD_LOG_NAME,
    )
)
