import torch

add_stat_value = torch._C._logging_add_stat_value
get_counters = torch._C._logging_get_counters
set_locking_logger = torch._C._logging_set_locking_logger