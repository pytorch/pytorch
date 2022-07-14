r'''
PyTorch Profiler is a tool that allows the collection of performance metrics during training and inference.
Profiler's context manager API can be used to better understand what model operators are the most expensive,
examine their input shapes and stack traces, study device kernel activity and visualize the execution trace.

.. note::
    An earlier version of the API in :mod:`torch.autograd` module is considered legacy and will be deprecated.

'''
from .profiler import profile, _KinetoProfile, \
    schedule, supported_activities, tensorboard_trace_handler, ProfilerAction, \
    _ExperimentalConfig, ExecutionGraphObserver
from torch._C._autograd import ProfilerActivity, kineto_available, _supported_activities, DeviceType
from torch.autograd.profiler import record_function

__all__ = ['profile', 'schedule', 'supported_activities',
           'tensorboard_trace_handler', 'ProfilerAction', 'ProfilerActivity',
           'kineto_available', 'DeviceType', 'record_function', 'ExecutionGraphObserver']

from . import itt
