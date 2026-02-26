from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class KernelTemplateParams(ABC):
    """Abstract base class for kernel template parameters."""

    @abstractmethod
    def to_kwargs(self) -> dict[str, Any]:
        """Convert params to kwargs dict for template.choice_or_none()"""

    @abstractmethod
    def to_serializeable_dict(self) -> dict[str, Any]:
        """Convert params to serializable dict for storage/caching"""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> KernelTemplateParams:
        """Create params instance from dict"""


class DictKernelTemplateParams(KernelTemplateParams):
    """Simple implementation that wraps a kwargs dict"""

    # NOTE: this is a compatibility layer, until every template
    # has time to define their own params class, with meaningful
    # defaults etc.

    def __init__(self, kwargs: dict[str, Any]):
        self.kwargs = kwargs

    def to_kwargs(self) -> dict[str, Any]:
        return self.kwargs.copy()

    def to_serializeable_dict(self) -> dict[str, Any]:
        return self.kwargs.copy()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DictKernelTemplateParams:
        return cls(data)
