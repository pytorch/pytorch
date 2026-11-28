from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.component_benchmark.component import Component
    from torch.utils.component_benchmark.result import BenchmarkResult, ComponentResult


class IProvider(ABC):
    """Adapt a training or inference framework to a module and real input data."""

    @abstractmethod
    def get_component(self) -> Component:
        raise NotImplementedError

    def get_submodule_filter(self) -> Callable[[str], bool] | None:
        return None

    def get_model_name(self) -> str:
        return self.get_component().name


class IPlugin(ABC):
    """Collect metrics or artifacts for an isolated component."""

    @abstractmethod
    def run(self, component: Component, copy_input: bool = False) -> ComponentResult:
        raise NotImplementedError


class IResultPlugin(ABC):
    """Consume structured results after all components have been processed."""

    @abstractmethod
    def run(self, result: BenchmarkResult) -> None:
        raise NotImplementedError
