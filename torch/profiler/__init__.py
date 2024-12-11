# mypy: allow-untyped-defs
r"""
PyTorch Profiler is a tool that allows the collection of performance metrics during training and inference.
Profiler's context manager API can be used to better understand what model operators are the most expensive,
examine their input shapes and stack traces, study device kernel activity and visualize the execution trace.

.. note::
    An earlier version of the API in :mod:`torch.autograd` module is considered legacy and will be deprecated.

"""
import os

from torch._C._autograd import _supported_activities, DeviceType, kineto_available
from torch._C._profiler import _ExperimentalConfig, ProfilerActivity, RecordScope
from torch._environment import is_fbcode
from torch.autograd.profiler import KinetoStepTracker, record_function
from torch.optim.optimizer import register_optimizer_step_post_hook

from .profiler import (
    _KinetoProfile,
    ExecutionTraceObserver,
    profile,
    ProfilerAction,
    schedule,
    supported_activities,
    tensorboard_trace_handler,
)


__all__ = [
    "profile",
    "schedule",
    "supported_activities",
    "tensorboard_trace_handler",
    "ProfilerAction",
    "ProfilerActivity",
    "kineto_available",
    "DeviceType",
    "record_function",
    "ExecutionTraceObserver",
]

from . import itt


def _optimizer_post_hook(optimizer, args, kwargs):
    KinetoStepTracker.increment_step("Optimizer")


if os.environ.get("KINETO_USE_DAEMON", "") or (
    is_fbcode() and os.environ.get("KINETO_FORCE_OPTIMIZER_HOOK", "")
):
    _ = register_optimizer_step_post_hook(_optimizer_post_hook)
