from __future__ import annotations

import copy
import gc
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, UTC
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
from torch.utils.component_benchmark.capture import (
    capture_components,
    get_submodules_with_depths,
)
from torch.utils.component_benchmark.component import (
    CallInput,
    Component,
    component_loss_and_bwd,
    default_loss_and_bwd,
)
from torch.utils.component_benchmark.interfaces import IPlugin, IProvider, IResultPlugin
from torch.utils.component_benchmark.result import (
    BenchmarkResult,
    ComponentResult,
    run_metadata,
)
from torch.utils.component_benchmark.timing import (
    CoreForwardBenchmarkPlugin,
    FlopsCounterPlugin,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _run_plugins(
    component: Component,
    plugins: Sequence[IPlugin],
    copy_input: bool,
) -> tuple[list[ComponentResult], dict[str, str]]:
    results = []
    errors = {}
    device = component.device
    cuda_devices = [device.index] if device.type == "cuda" else []
    for plugin in plugins:
        name = getattr(plugin, "label", type(plugin).__name__)
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                result = plugin.run(copy.deepcopy(component), copy_input=copy_input)
            results.append(result)
        except Exception:
            errors[name] = traceback.format_exc()
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results, errors


class PyTorchProvider(IProvider):
    """Provide an ordinary PyTorch module and explicit positional/keyword inputs."""

    def __init__(
        self,
        model: nn.Module,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        name: str = "model",
    ) -> None:
        self.component = Component(
            model, [CallInput(args=args, kwargs=kwargs or {})], name=name
        )

    def get_component(self) -> Component:
        return self.component


class ComponentBenchmark:
    """Capture real submodule inputs and run isolated profiler plugins."""

    def __init__(
        self,
        component_provider: IProvider,
        custom_plugins: Sequence[IPlugin] | None = None,
        result_plugins: Sequence[IResultPlugin] = (),
    ) -> None:
        self.provider = component_provider
        self.plugins = list(custom_plugins) if custom_plugins is not None else None
        self.result_plugins = list(result_plugins)

    def list_subcomponents(
        self, max_depth: int = 3, submodule_path: str = ""
    ) -> list[str]:
        model = self.provider.get_component().model
        return get_submodules_with_depths(
            model.get_submodule(submodule_path), max_depth, submodule_path
        )

    def get_subcomponent(
        self, submodule_path: str, *, capture_output_grad: bool = False
    ) -> Component:
        return capture_components(
            self.provider.get_component(),
            [submodule_path],
            capture_output_grad=capture_output_grad,
        )[submodule_path]

    def create_model_graph(self, submodule_path: str = "") -> str:
        """Return an FX graph as text, suitable for a local artifact."""
        model = self.provider.get_component().model.get_submodule(submodule_path)
        return str(torch.fx.symbolic_trace(model).graph)

    def run(
        self,
        *,
        max_depth: int = 3,
        submodule_path: str = "",
        submodule_filter: Callable[[str], bool] | None = None,
        copy_input: bool = True,
        capture_output_grad: bool = False,
        loss_and_bwd: Callable[[Any], None] | str | bool | None = None,
        use_multiprocess: bool = False,
    ) -> BenchmarkResult:
        """Benchmark active modules; isolate each plugin failure.

        Multiprocess mode uses one spawned worker to avoid concurrent benchmark
        interference. Models, providers' inputs and plugins must be picklable.
        An empty ``custom_plugins`` list performs capture/discovery only.
        """
        component = self.provider.get_component()
        if loss_and_bwd is True or loss_and_bwd == "component":
            loss_and_bwd = component_loss_and_bwd
        elif loss_and_bwd == "default":
            loss_and_bwd = default_loss_and_bwd
        elif loss_and_bwd is False:
            loss_and_bwd = None
        elif isinstance(loss_and_bwd, str):
            raise ValueError(f"Unknown backward mode: {loss_and_bwd}")
        plugins = (
            self.plugins
            if self.plugins is not None
            else [
                CoreForwardBenchmarkPlugin(loss_and_bwd=loss_and_bwd),
                FlopsCounterPlugin(loss_and_bwd=loss_and_bwd),
            ]
        )
        paths = get_submodules_with_depths(
            component.model.get_submodule(submodule_path), max_depth, submodule_path
        )
        filters = [self.provider.get_submodule_filter(), submodule_filter]
        paths = [
            path
            for path in paths
            if all(predicate is None or predicate(path) for predicate in filters)
        ]
        result = BenchmarkResult(
            metadata={
                **run_metadata(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device": str(component.device),
                "max_depth": max_depth,
                "submodule_path": submodule_path,
                "copy_input": copy_input,
                "capture_output_grad": capture_output_grad,
                "execution": "serial spawned worker"
                if use_multiprocess
                else "serial in process",
            }
        )
        components = capture_components(
            component,
            paths,
            capture_output_grad=capture_output_grad,
            loss_and_bwd=loss_and_bwd or component_loss_and_bwd,
        )
        executor = (
            ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
            if use_multiprocess
            else None
        )
        try:
            for captured in components.values():
                name = captured.name
                if not captured.data:
                    status = (
                        "container"
                        if type(captured.model)
                        in (
                            nn.ModuleList,
                            nn.ModuleDict,
                            nn.ParameterList,
                            nn.ParameterDict,
                        )
                        else "inactive"
                    )
                    result.add_component_metrics(
                        name, ComponentResult({"status": status, "invocation_count": 0})
                    )
                    continue
                result.add_component_metrics(
                    name,
                    ComponentResult(
                        {
                            "status": "captured",
                            "invocation_count": captured.get_iteration_call_count(),
                        }
                    ),
                )
                if executor is not None:
                    plugin_results, errors = executor.submit(
                        _run_plugins, captured, plugins, copy_input
                    ).result()
                else:
                    plugin_results, errors = _run_plugins(captured, plugins, copy_input)
                for plugin_result in plugin_results:
                    try:
                        result.add_component_metrics(name, plugin_result)
                    except ValueError:
                        errors[plugin_result.plugin_name or "metrics"] = (
                            traceback.format_exc()
                        )
                for plugin, error in errors.items():
                    result.add_error(name, error, plugin)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        result.metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
        for plugin in self.result_plugins:
            plugin.run(result)
        return result


def run_cb_with_model_provider(
    model_provider: IProvider,
    *,
    custom_plugins: Sequence[IPlugin] | None = None,
    result_plugins: Sequence[IResultPlugin] = (),
    **kwargs: Any,
) -> BenchmarkResult:
    return ComponentBenchmark(model_provider, custom_plugins, result_plugins).run(
        **kwargs
    )


def run_cb(
    model: nn.Module,
    model_input: Any,
    *,
    model_kwargs: dict[str, Any] | None = None,
    custom_plugins: Sequence[IPlugin] | None = None,
    result_plugins: Sequence[IResultPlugin] = (),
    **kwargs: Any,
) -> BenchmarkResult:
    """Benchmark a model, accepting one input or a tuple of positional inputs."""
    args = model_input if isinstance(model_input, tuple) else (model_input,)
    provider = PyTorchProvider(model, args, model_kwargs)
    return run_cb_with_model_provider(
        provider, custom_plugins=custom_plugins, result_plugins=result_plugins, **kwargs
    )
