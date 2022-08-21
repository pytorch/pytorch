from enum import Enum
from typing import List, Union

# defined in torch/csrc/profiler/python/init.cpp

class ProfilerState(Enum):
    Disable = ...
    CPU = ...
    CUDA = ...
    NVTX = ...
    ITT = ...
    KINETO = ...
    KINETO_GPU_FALLBACK = ...

class ActiveProfilerType:
    ...

class ProfilerActivity(Enum):
    CPU = ...
    CUDA = ...

class _ExperimentalConfig:
    def __init__(
        self,
        profiler_metrics: List[str] = ...,
        profiler_measure_per_kernel: bool = ...,
    ) -> None: ...
    ...

class ProfilerConfig:
    def __init__(
        self,
        state: ProfilerState,
        report_input_shapes: bool,
        profile_memory: bool,
        with_stack: bool,
        with_flops: bool,
        with_modules: bool,
        experimental_config: _ExperimentalConfig,
    ) -> None: ...
    ...

class _ProfilerEvent:
    tag: _EventType
    id: int
    correlation_id: int
    start_tid: int
    start_time_ns: int
    end_time_ns: int
    duration_time_ns: int
    parent: _ProfilerEvent
    children: List[_ProfilerEvent]
    extra_fields: Union[_ExtraFields_Allocation, _ExtraFields_Backend,
                        _ExtraFields_PyCall, _ExtraFields_PyCCall,
                        _ExtraFields_TorchOp]
    def name(self) -> str: ...
    ...

class _PyFrameState:
    line_number: int
    function_name: str
    file_name: str
    ...

class _EventType(Enum):
    Allocation = ...
    Backend = ...
    PyCall = ...
    PyCCall = ...
    TorchOp = ...
    Kineto = ...

class _Inputs:
    shapes: List[List[int]]
    dtypes: List[str]

class _ExtraFields_TorchOp:
    allow_tf32_cublas: bool
    inputs: _Inputs
    ...

class _ExtraFields_Backend:
    ...

class _ExtraFields_Allocation:
    ...

class _ExtraFields_PyCCall:
    caller: _PyFrameState
    ...

class _ExtraFields_PyCall:
    caller: _PyFrameState
    ...

class _ExtraFields_Kineto:
    ...
