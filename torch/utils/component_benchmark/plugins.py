from __future__ import annotations

import copy
import json
import math
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.component_benchmark.execution import Iteration
from torch.utils.component_benchmark.interfaces import IPlugin, IResultPlugin
from torch.utils.component_benchmark.result import ComponentResult
from torch.utils.component_benchmark.timing import _flops, _synchronize, _timing


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn
    from torch.utils.component_benchmark.component import Component
    from torch.utils.component_benchmark.result import BenchmarkResult


class _TensorIOMode(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.bytes = 0
        self.operators: dict[str, int] = {}

    def __torch_dispatch__(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        output = func(*args, **(kwargs or {}))
        if not func._schema.is_mutable and any(
            result.alias_info is not None for result in func._schema.returns
        ):
            return output
        size = sum(
            value.numel() * value.element_size()
            for value in tree_leaves((args, kwargs, output))
            if isinstance(value, torch.Tensor)
        )
        name = str(func)
        self.bytes += size
        self.operators[name] = self.operators.get(name, 0) + size
        return output


class TorchDispatchModesPlugin(IPlugin):
    """Estimate eager tensor input/output traffic, not physical memory traffic.

    Read-only alias/view operations are excluded. The estimate counts tensor
    elements passed to and returned from each remaining operation. Caches,
    fusion and kernel-specific memory access patterns can change actual traffic.
    """

    def __init__(
        self,
        *,
        peak_bandwidth_bytes_per_second: float | None = None,
        min_run_time: float = 0.2,
        loss_and_bwd: Callable[[Any], None] | None = None,
    ) -> None:
        if peak_bandwidth_bytes_per_second is not None and (
            not math.isfinite(peak_bandwidth_bytes_per_second)
            or peak_bandwidth_bytes_per_second <= 0
        ):
            raise ValueError(
                "peak_bandwidth_bytes_per_second must be positive and finite"
            )
        self.peak = peak_bandwidth_bytes_per_second
        self.min_run_time = min_run_time
        self.loss_and_bwd = loss_and_bwd

    def run(self, component: Component, copy_input: bool = False) -> ComponentResult:
        iteration = Iteration(
            component, copy_input=copy_input, loss_and_bwd=self.loss_and_bwd
        )
        with _TensorIOMode() as mode:
            iteration()
        timing = _timing(iteration, 5, self.min_run_time)
        bandwidth = mode.bytes / (timing["median_ms"] / 1000)
        metrics = {
            **timing,
            "estimated_tensor_io_bytes": mode.bytes,
            "estimated_bandwidth_bytes_per_second": bandwidth,
            "operator_tensor_io_bytes": mode.operators,
            "scope": "one eager replay's logical tensor IO; not measured device memory traffic",
        }
        if self.peak is not None:
            metrics["peak_bandwidth_bytes_per_second"] = self.peak
            metrics["estimated_utilization_percent"] = 100 * bandwidth / self.peak
        return ComponentResult(metrics, plugin_name="tensor_io")


class MemoryPlugin(IPlugin):
    """Measure CUDA allocator peaks; CPU allocator peaks are unavailable.

    Optional memory history uses PyTorch's CUDA memory snapshot API. Snapshots
    are saved locally and can be opened with the PyTorch memory visualizer.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        loss_and_bwd: Callable[[Any], None] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.loss_and_bwd = loss_and_bwd

    def run(self, component: Component, copy_input: bool = False) -> ComponentResult:
        iteration = Iteration(
            component, copy_input=copy_input, loss_and_bwd=self.loss_and_bwd
        )
        device = component.device
        if device.type != "cuda":
            return ComponentResult(
                {"peak_allocated_bytes": None, "scope": "CUDA allocator only"},
                plugin_name="memory",
            )
        _synchronize(device)
        artifacts = {}
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            if self.output_dir is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                torch.cuda.memory._record_memory_history(max_entries=100000)
            try:
                iteration()
                _synchronize(device)
                metrics = {
                    "baseline_allocated_bytes": baseline,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "incremental_peak_bytes": torch.cuda.max_memory_allocated(device)
                    - baseline,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
                if self.output_dir is not None:
                    path = self.output_dir / f"memory_{uuid.uuid4().hex}.pickle"
                    torch.cuda.memory._dump_snapshot(str(path))
                    artifacts["snapshot"] = str(path.resolve())
            finally:
                if self.output_dir is not None:
                    torch.cuda.memory._record_memory_history(enabled=None)
        return ComponentResult(metrics, plugin_name="memory", artifacts=artifacts)


class ProfilerPlugin(IPlugin):
    """Save a local Kineto trace and operator/kernel summaries."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        iterations: int = 3,
        loss_and_bwd: Callable[[Any], None] | None = None,
        torch_compile: bool = False,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        self.output_dir = Path(output_dir)
        self.iterations = iterations
        self.loss_and_bwd = loss_and_bwd
        self.torch_compile = torch_compile

    def run(self, component: Component, copy_input: bool = False) -> ComponentResult:
        iteration = Iteration(
            component,
            copy_input=copy_input,
            loss_and_bwd=self.loss_and_bwd,
            torch_compile=self.torch_compile,
        )
        iteration()
        _synchronize(component.device)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if component.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True,
        ) as profiler:
            for _ in range(self.iterations):
                iteration()
                profiler.step()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        trace_path = self.output_dir / f"trace_{token}.json"
        summary_path = self.output_dir / f"operators_{token}.json"
        profiler.export_chrome_trace(str(trace_path))
        operators = [
            {
                "name": event.key,
                "count": event.count,
                "self_cpu_time_total_us": event.self_cpu_time_total,
                "self_device_time_total_us": event.self_device_time_total,
                "flops": event.flops,
                "input_shapes": event.input_shapes,
            }
            for event in profiler.key_averages(group_by_input_shape=True)
        ]
        summary_path.write_text(json.dumps(operators, indent=2), encoding="utf-8")
        return ComponentResult(
            {
                "iterations": self.iterations,
                "self_cpu_time_total_us": sum(
                    event["self_cpu_time_total_us"] for event in operators
                ),
                "self_device_time_total_us": sum(
                    event["self_device_time_total_us"] for event in operators
                ),
            },
            plugin_name="profiler",
            artifacts={
                "trace": str(trace_path.resolve()),
                "operators": str(summary_path.resolve()),
            },
        )


class MFUPlugin(IPlugin):
    """Compute formula-covered FLOP utilization using a caller-supplied peak.

    The peak must match the device, precision and sparsity of this workload.
    No hardware limits are guessed. FLOPs and wall latency use the same eager
    iteration definition; the result includes CPU overhead and copying.
    """

    def __init__(
        self,
        peak_flops_per_second: float,
        *,
        min_run_time: float = 0.2,
        loss_and_bwd: Callable[[Any], None] | None = None,
    ) -> None:
        if not math.isfinite(peak_flops_per_second) or peak_flops_per_second <= 0:
            raise ValueError("peak_flops_per_second must be positive and finite")
        self.peak = peak_flops_per_second
        self.min_run_time = min_run_time
        self.loss_and_bwd = loss_and_bwd

    def run(self, component: Component, copy_input: bool = False) -> ComponentResult:
        iteration = Iteration(
            component, copy_input=copy_input, loss_and_bwd=self.loss_and_bwd
        )
        flops = _flops(iteration)
        timing = _timing(iteration, 5, self.min_run_time)
        achieved = flops["total"] / (timing["median_ms"] / 1000)
        return ComponentResult(
            {
                **timing,
                "flops": flops["total"],
                "peak_flops_per_second": self.peak,
                "achieved_flops_per_second": achieved,
                "percent": 100 * achieved / self.peak,
                "flops_scope": flops["scope"],
            },
            plugin_name="mfu",
        )


class AccuracyCheckPlugin(IPlugin):
    """Compare eager and transformed outputs, gradients, parameters and buffers."""

    def __init__(
        self,
        *,
        torch_compile: bool = True,
        compile_options: dict[str, Any] | None = None,
        preproc_func: Callable[[Component], Component] | None = None,
        loss_and_bwd: Callable[[Any], None] | None = None,
        optimizer_creator_fn: Callable[[nn.Module, torch.device], torch.optim.Optimizer]
        | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> None:
        self.torch_compile = torch_compile
        self.compile_options = compile_options
        self.preproc_func = preproc_func
        self.loss_and_bwd = loss_and_bwd
        self.optimizer_creator_fn = optimizer_creator_fn
        self.rtol, self.atol = rtol, atol

    def run(self, component: Component, copy_input: bool = False) -> ComponentResult:
        reference = Iteration(
            component,
            copy_input=True,
            loss_and_bwd=self.loss_and_bwd,
            optimizer_creator_fn=self.optimizer_creator_fn,
        )
        transformed = (
            self.preproc_func(copy.deepcopy(component))
            if self.preproc_func
            else component
        )
        actual = Iteration(
            transformed,
            copy_input=True,
            loss_and_bwd=self.loss_and_bwd,
            torch_compile=self.torch_compile,
            compile_options=self.compile_options,
            optimizer_creator_fn=self.optimizer_creator_fn,
        )
        cuda_devices = (
            [component.device.index] if component.device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            cpu_rng = torch.get_rng_state()
            gpu_rng = (
                torch.cuda.get_rng_state(component.device) if cuda_devices else None
            )
            expected_output = reference()
            torch.set_rng_state(cpu_rng)
            if gpu_rng is not None:
                torch.cuda.set_rng_state(gpu_rng, component.device)
            actual_output = actual()
        pairs = {
            "output": (actual_output, expected_output),
            "parameters": (
                dict(actual.model.named_parameters()),
                dict(reference.model.named_parameters()),
            ),
            "gradients": (
                {name: value.grad for name, value in actual.model.named_parameters()},
                {
                    name: value.grad
                    for name, value in reference.model.named_parameters()
                },
            ),
            "buffers": (
                dict(actual.model.named_buffers()),
                dict(reference.model.named_buffers()),
            ),
        }
        if actual.optimizer is not None and reference.optimizer is not None:
            pairs["optimizer"] = (
                actual.optimizer.state_dict(),
                reference.optimizer.state_dict(),
            )
        metrics = {}
        for label, (value, expected) in pairs.items():
            torch.testing.assert_close(
                value, expected, rtol=self.rtol, atol=self.atol, check_dtype=False
            )
            metrics[label] = True
        return ComponentResult(metrics, plugin_name="accuracy")


class GPUBoundSimulationPlugin(IPlugin):
    """Measure CUDA graph replay, excluding Python launch and input-copy costs.

    Inputs must be graph-safe and non-mutating. Optimizers and preprocessing
    are intentionally supplied through custom plugins for graph-specific work.
    """

    def __init__(
        self,
        *,
        iterations: int = 100,
        loss_and_bwd: Callable[[Any], None] | None = None,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        self.iterations = iterations
        self.loss_and_bwd = loss_and_bwd

    def run(self, component: Component, copy_input: bool = False) -> ComponentResult:
        if component.device.type != "cuda":
            raise ValueError("CUDA graph replay requires a CUDA component")
        if copy_input:
            raise ValueError(
                "CUDA graph replay requires copy_input=False and non-mutating inputs"
            )
        iteration = Iteration(component, loss_and_bwd=self.loss_and_bwd)
        with torch.cuda.device(component.device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    iteration()
            torch.cuda.current_stream().wait_stream(stream)
            iteration.model.zero_grad(set_to_none=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                iteration()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(self.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            milliseconds = start.elapsed_time(end) / self.iterations
        return ComponentResult(
            {
                "mean_ms": milliseconds,
                "iterations": self.iterations,
                "scope": "CUDA event time for graph replay",
            },
            plugin_name="cuda_graph",
        )


class ComponentCapturePlugin(IPlugin):
    """Save isolated components to trusted local replay files."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def run(self, component: Component, copy_input: bool = False) -> ComponentResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"component_{uuid.uuid4().hex}.pt"
        component.save(path)
        return ComponentResult(
            {}, plugin_name="capture", artifacts={"component": str(path.resolve())}
        )


class ReportPlugin(IResultPlugin):
    """Write local JSON, CSV and interactive HTML reports."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def run(self, result: BenchmarkResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result.save_json(self.output_dir / "results.json")
        result.save_csv(self.output_dir / "results.csv")
        result.save_html(self.output_dir / "report.html")
