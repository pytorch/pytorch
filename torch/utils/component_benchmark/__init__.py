"""Independent hierarchical benchmarking of PyTorch modules."""

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
from torch.utils.component_benchmark.preprocessing import (
    compose,
    repeat_batch,
    to_dtype,
)
from torch.utils.component_benchmark.result import BenchmarkResult, ComponentResult
