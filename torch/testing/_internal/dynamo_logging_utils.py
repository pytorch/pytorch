import torch._logging.loggable_types
import functools
from . import logging_utils

make_test = functools.partial(
    logging_utils.make_test,
    log_names=(
        torch._logging.loggable_types.TORCHDYNAMO_LOG_NAME,
        torch._logging.loggable_types.TORCHINDUCTOR_LOG_NAME,
        torch._logging.loggable_types.AOT_AUTOGRAD_LOG_NAME,
    )
)
