import functools

import torch

# to lower notebook execution time while hiding backend="eager"
torch.compile = functools.partial(torch.compile, backend="eager")

# to clear torch logs format
import os

os.environ["TORCH_LOGS_FORMAT"] = ""
torch._logging._internal._init_logs()
