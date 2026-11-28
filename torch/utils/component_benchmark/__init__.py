"""Independent hierarchical benchmarking of PyTorch modules."""

from torch.utils.component_benchmark.benchmark import (
    ComponentBenchmark,
    PyTorchProvider,
    run_cb,
    run_cb_with_model_provider,
)
from torch.utils.component_benchmark.capture import (
    capture_components,
    get_submodules_with_depths,
)
from torch.utils.component_benchmark.component import (
    CallInput,
    Component,
    component_loss_and_bwd,
    copy_data,
    default_loss_and_bwd,
)
from torch.utils.component_benchmark.interfaces import IPlugin, IProvider, IResultPlugin
from torch.utils.component_benchmark.plugins import (
    AccuracyCheckPlugin,
    ComponentCapturePlugin,
    GPUBoundSimulationPlugin,
    MemoryPlugin,
    MFUPlugin,
    ProfilerPlugin,
    ReportPlugin,
    TorchDispatchModesPlugin,
)
from torch.utils.component_benchmark.preprocessing import (
    compose,
    repeat_batch,
    to_dtype,
)
from torch.utils.component_benchmark.result import BenchmarkResult, ComponentResult
from torch.utils.component_benchmark.timing import (
    CoreForwardBenchmarkPlugin,
    FlopsCounterPlugin,
)


__all__ = [
    "AccuracyCheckPlugin",
    "BenchmarkResult",
    "CallInput",
    "Component",
    "ComponentBenchmark",
    "ComponentCapturePlugin",
    "ComponentResult",
    "CoreForwardBenchmarkPlugin",
    "FlopsCounterPlugin",
    "GPUBoundSimulationPlugin",
    "IPlugin",
    "IProvider",
    "IResultPlugin",
    "MFUPlugin",
    "MemoryPlugin",
    "ProfilerPlugin",
    "PyTorchProvider",
    "ReportPlugin",
    "TorchDispatchModesPlugin",
    "capture_components",
    "component_loss_and_bwd",
    "compose",
    "copy_data",
    "default_loss_and_bwd",
    "get_submodules_with_depths",
    "repeat_batch",
    "run_cb",
    "run_cb_with_model_provider",
    "to_dtype",
]
