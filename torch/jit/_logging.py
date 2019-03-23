import torch

add_stat_value = torch.ops.prim.BumpCounter
get_counter_val = torch.ops.prim.GetCounter

set_logger = torch._C._logging_set_logger
LockingLogger = torch._C.LockingLogger
AggregationType = torch._C.AggregationType
NoopLogger = torch._C.NoopLogger

time_point = torch.ops.prim.TimePoint
