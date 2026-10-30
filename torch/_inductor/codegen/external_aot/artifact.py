from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXTERNAL_AOT_SIDECAR_VERSION = 1


@dataclass(frozen=True)
class ExternalAOTTensorArg:
    name: str
    source: str
    dynamic_sizes: tuple[int, ...] = ()
    dynamic_strides: tuple[int, ...] = ()
    read_only: bool = True
    mutated: bool = False
    workspace: bool = False
    alias_of: str | None = None
    flattened: bool = False


@dataclass(frozen=True)
class ExternalAOTScalarArg:
    name: str
    ctype: str
    source: str


@dataclass(frozen=True)
class ExternalAOTBuildSpec:
    kind: str
    kernel_name: str
    target: str
    source_hash: str
    tensor_args: tuple[ExternalAOTTensorArg, ...]
    scalar_args: tuple[ExternalAOTScalarArg, ...] = ()
    options: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "kernel_name": self.kernel_name,
            "target": self.target,
            "source_hash": self.source_hash,
            "tensor_args": [asdict(arg) for arg in self.tensor_args],
            "scalar_args": [asdict(arg) for arg in self.scalar_args],
            "options": list(self.options),
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExternalAOTArtifact:
    kind: str
    object_path: str
    header_path: str
    sidecar_path: str
    content_hash: str
    target: str
    link_libraries: tuple[str, ...]
    link_options: tuple[str, ...]
    payload: dict[str, Any]

    @classmethod
    def from_files(
        cls,
        *,
        spec: ExternalAOTBuildSpec,
        object_path: str | os.PathLike[str],
        header_path: str | os.PathLike[str],
        link_libraries: tuple[str, ...],
        link_options: tuple[str, ...],
        payload: dict[str, Any],
        runtime_versions: dict[str, str | None],
    ) -> ExternalAOTArtifact:
        object_path = Path(object_path)
        header_path = Path(header_path)
        object_hash = hashlib.sha256(object_path.read_bytes()).hexdigest()
        header_hash = hashlib.sha256(header_path.read_bytes()).hexdigest()
        artifact_hash = hashlib.sha256(
            json.dumps(
                {
                    "spec": spec.to_dict(),
                    "object_hash": object_hash,
                    "header_hash": header_hash,
                    "link_libraries": link_libraries,
                    "link_options": link_options,
                    "payload": payload,
                    "runtime_versions": runtime_versions,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        sidecar_path = object_path.with_suffix(".json")
        sidecar = {
            "version": EXTERNAL_AOT_SIDECAR_VERSION,
            "content_hash": artifact_hash,
            "kind": spec.kind,
            "spec": spec.to_dict(),
            "target": spec.target,
            "runtime_versions": runtime_versions,
            "files": {
                "object": {
                    "path": object_path.name,
                    "sha256": object_hash,
                    "size": object_path.stat().st_size,
                },
                "header": {
                    "path": header_path.name,
                    "sha256": header_hash,
                    "size": header_path.stat().st_size,
                },
            },
            "link": {
                "libraries": list(link_libraries),
                "options": list(link_options),
            },
            "payload": payload,
        }
        _write_sidecar_atomically(sidecar_path, sidecar)
        return cls(
            kind=spec.kind,
            object_path=str(object_path),
            header_path=str(header_path),
            sidecar_path=str(sidecar_path),
            content_hash=artifact_hash,
            target=spec.target,
            link_libraries=link_libraries,
            link_options=link_options,
            payload=payload,
        )

    def validate_files(self) -> None:
        sidecar = json.loads(Path(self.sidecar_path).read_text())
        if sidecar.get("version") != EXTERNAL_AOT_SIDECAR_VERSION:
            raise RuntimeError(
                f"unsupported external AOT sidecar version: {sidecar.get('version')!r}"
            )
        for kind, path in (
            ("object", Path(self.object_path)),
            ("header", Path(self.header_path)),
        ):
            expected = sidecar["files"][kind]["sha256"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"stale external AOT {kind}: expected {expected}, got {actual}"
                )


def _write_sidecar_atomically(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(value, file, sort_keys=True, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def publish_external_aot_artifact(
    wrapper,
    kernel_name: str,
    artifact: ExternalAOTArtifact,
    deferred_call: Any,
) -> ExternalAOTArtifact:
    artifact.validate_files()
    cache_key = (artifact.kind, artifact.target, artifact.content_hash)
    artifact = wrapper.external_aot_artifact_cache.setdefault(cache_key, artifact)
    wrapper.external_aot_artifacts[kernel_name] = artifact
    wrapper.deferred_aot_call_wrappers[kernel_name] = deferred_call
    wrapper.aot_kernel_objects.add(artifact.object_path)
    wrapper.aot_link_libraries.update(artifact.link_libraries)
    wrapper.aot_link_options.update(artifact.link_options)
    for path in (artifact.header_path, artifact.sidecar_path):
        if path not in wrapper.additional_files:
            wrapper.additional_files.append(path)
    return artifact
