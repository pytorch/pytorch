import torch

add_stat_value = torch._C._logging_add_stat_value
get_counters = torch._C._logging_get_counters

set_logger = torch._C._logging_set_logger
LockingLogger = torch._C.LockingLogger
NoopLogger = torch._C.NoopLogger

time_point = torch._C._logging_time_point
record_duration_since = torch._C._logging_record_duration_since