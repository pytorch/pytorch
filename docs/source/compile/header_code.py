import functools
import os

import torch


# to lower notebook execution time while hiding backend="eager"
torch.compile = functools.partial(torch.compile, backend="eager")

# to clear torch logs format
os.environ["TORCH_LOGS_FORMAT"] = ""
torch._logging._internal.DEFAULT_FORMATTER = (
    torch._logging._internal._default_formatter()
)
torch._logging._internal._init_logs()
