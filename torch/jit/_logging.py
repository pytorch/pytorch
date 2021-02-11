import torch

add_stat_value = torch.ops.prim.AddStatValue

set_logger = torch._C._jit._logging_set_logger
LockingLogger = torch._C._jit.LockingLogger
AggregationType = torch._C._jit.AggregationType
NoopLogger = torch._C._jit.NoopLogger

time_point = torch.ops.prim.TimePoint
