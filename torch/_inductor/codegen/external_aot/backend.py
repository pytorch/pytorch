from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .artifact import ExternalAOTArtifact, ExternalAOTBuildSpec


class ExternalAOTBackend(ABC):
    """Backend adapter for a DSL that exports linkable C objects."""

    kind: str

    @abstractmethod
    def export(
        self,
        request: Any,
        spec: ExternalAOTBuildSpec,
        output_dir: str,
    ) -> ExternalAOTArtifact:
        raise NotImplementedError

    @abstractmethod
    def validate_abi(
        self,
        artifact: ExternalAOTArtifact,
        spec: ExternalAOTBuildSpec,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_deferred_call(
        self,
        artifact: ExternalAOTArtifact,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def emit_call(
        self,
        wrapper,
        artifact: ExternalAOTArtifact,
        binding: Any,
        *,
        device: Any,
    ) -> None:
        raise NotImplementedError


_BACKENDS: dict[str, ExternalAOTBackend] = {}


def register_external_aot_backend(backend: ExternalAOTBackend) -> None:
    previous = _BACKENDS.get(backend.kind)
    if previous is not None and type(previous) is not type(backend):
        raise RuntimeError(
            f"external AOT backend {backend.kind!r} is already registered"
        )
    _BACKENDS[backend.kind] = backend


def get_external_aot_backend(kind: str) -> ExternalAOTBackend:
    try:
        return _BACKENDS[kind]
    except KeyError as error:
        raise RuntimeError(
            f"external AOT backend {kind!r} is not registered"
        ) from error
