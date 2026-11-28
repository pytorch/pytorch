from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
from torch.utils.benchmark import Timer
from torch.utils.component_benchmark.execution import Iteration
from torch.utils.component_benchmark.interfaces import IPlugin
from torch.utils.component_benchmark.result import ComponentResult
from torch.utils.flop_counter import FlopCounterMode


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.component_benchmark.component import Component


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type != "cpu":
        raise ValueError(f"Unsupported timing device: {device}")


def _timing(iteration: Iteration, warmup: int, min_run_time: float) -> dict[str, Any]:
    if warmup < 0 or not math.isfinite(min_run_time) or min_run_time <= 0:
        raise ValueError("warmup must be nonnegative and min_run_time must be positive")
    iteration.warmup()
    for _ in range(warmup):
        iteration()
    _synchronize(iteration.device)

    def run() -> None:
        iteration()
        _synchronize(iteration.device)

    measurement = Timer(
        stmt="run()", globals={"run": run}, num_threads=torch.get_num_threads()
    ).blocked_autorange(min_run_time=min_run_time)
    return {
        "median_ms": measurement.median * 1000,
        "mean_ms": measurement.mean * 1000,
        "iqr_ms": measurement.iqr * 1000,
        "samples_ms": [value * 1000 for value in measurement.times],
        "number_per_run": measurement.number_per_run,
        "sample_count": len(measurement.times),
        "cpu_threads": measurement.task_spec.num_threads,
        "timing_scope": "synchronized wall time; input copying included when requested",
    }


def _flops(iteration: Iteration) -> dict[str, Any]:
    with FlopCounterMode(display=False) as counter:
        iteration()
    return {
        "total": counter.get_total_flops(),
        "operators": {
            str(op): count
            for op, count in counter.get_flop_counts().get("Global", {}).items()
        },
        "scope": "registered FlopCounterMode formulas; unsupported operators are not counted",
    }


class CoreForwardBenchmarkPlugin(IPlugin):
    """Measure eager or compiled forward/backward/optimizer iteration latency."""

    def __init__(
        self,
        *,
        warmup_iterations: int = 5,
        min_run_time: float = 0.2,
        torch_compile: bool = False,
        compile_options: dict[str, Any] | None = None,
        loss_and_bwd: Callable[[Any], None] | None = None,
        optimizer_creator_fn: Callable[[nn.Module, torch.device], torch.optim.Optimizer]
        | None = None,
        activation_checkpointing: bool = False,
        preproc_func: Callable[[Component], Component] | None = None,
        label: str | None = None,
    ) -> None:
        self.warmup = warmup_iterations
        self.min_run_time = min_run_time
        self.preproc_func = preproc_func
        self.label = label or ("compiled" if torch_compile else "eager")
        self.options = {
            "torch_compile": torch_compile,
            "compile_options": compile_options,
            "loss_and_bwd": loss_and_bwd,
            "optimizer_creator_fn": optimizer_creator_fn,
            "activation_checkpointing": activation_checkpointing,
        }

    def run(self, component: Component, copy_input: bool = False) -> ComponentResult:
        if self.preproc_func is not None:
            component = self.preproc_func(component)
        iteration = Iteration(component, copy_input=copy_input, **self.options)
        metrics = _timing(iteration, self.warmup, self.min_run_time)
        metrics["invocation_count"] = component.get_iteration_call_count()
        metrics["model_bytes"] = sum(
            parameter.numel() * parameter.element_size()
            for parameter in component.model.parameters()
        )
        metrics["backward"] = iteration.backward
        metrics["optimizer_step"] = iteration.optimizer is not None
        return ComponentResult(metrics, plugin_name=self.label)


class FlopsCounterPlugin(IPlugin):
    """Count formula-covered eager FLOPs, optionally including backward."""

    def __init__(self, loss_and_bwd: Callable[[Any], None] | None = None) -> None:
        self.loss_and_bwd = loss_and_bwd

    def run(self, component: Component, copy_input: bool = False) -> ComponentResult:
        iteration = Iteration(
            component, copy_input=copy_input, loss_and_bwd=self.loss_and_bwd
        )
        return ComponentResult(_flops(iteration), plugin_name="flops")
