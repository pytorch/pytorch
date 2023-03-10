import torch._dynamo.logging as td_logging
import unittest
import functools
from . import logging_utils

# This is needed because we reinit logging each time dynamo is called
def patch_handlers_after_init_logging(patch_handlers_fn):
    old_init_logging = td_logging.init_logging

    def new_init_logging(log_file_name=None):
        old_init_logging(log_file_name)
        patch_handlers_fn()

    return unittest.mock.patch.object(td_logging, "init_logging", new_init_logging)


make_test = functools.partial(
    logging_utils.make_test,
    log_names=(
        td_logging.TORCHDYNAMO_LOG_NAME,
        td_logging.TORCHINDUCTOR_LOG_NAME,
        td_logging.AOT_AUTOGRAD_LOG_NAME,
    ),
    install_hook=patch_handlers_after_init_logging,
)
