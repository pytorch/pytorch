"""Runtime dependency transport for precompiled workers."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import pickle
import threading
from collections.abc import Iterator
from types import ModuleType
from typing import cast, TYPE_CHECKING

from torch.compiler._cache import (
    _serialize_single_cache,
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
)
from torch.compiler._no_compile import is_compilation_forbidden, no_compilation
from torch.utils._appending_byte_serializer import AppendingByteSerializer


if TYPE_CHECKING:
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner


_frozen_triton_kernels: dict[str, bytes] = {}
_frozen_triton_kernels_lock = threading.Lock()
_frozen_cpp_kernels: dict[str, bytes] = {}
_frozen_cpp_kernels_lock = threading.Lock()


def _triton_runtime_cache() -> ModuleType | None:
    """Return ``triton.runtime.cache`` if this Triton can export its runtime cache.

    Without it, finalization still freezes Inductor's static Triton launchers and
    C++ kernels, but kernels launched directly through Triton's JIT are not captured
    and compile again on first use in the serving process.
    """
    try:
        from triton.runtime import cache
    except ImportError:
        return None
    return cache if hasattr(cache, "export_runtime_cache") else None


def _runtime_context() -> dict[str, str | None]:
    import torch
    from torch._inductor.codecache import torch_key

    return {"torch_key": torch_key().hex(), "torch_cuda": torch.version.cuda}


@CacheArtifactFactory.register
class TritonRuntimeCacheArtifact(CacheArtifact):
    @staticmethod
    def type() -> str:
        return "triton_runtime"

    def populate_cache(self) -> None:
        if hashlib.sha256(self.content).hexdigest() != self.key:
            raise RuntimeError("Corrupt Triton runtime-cache artifact")
        cache = _triton_runtime_cache()
        if cache is None:
            raise RuntimeError(
                "The precompile cache contains a Triton runtime cache, but the "
                "installed Triton cannot import one"
            )
        cache.import_runtime_cache(self.content, context=_runtime_context())


@CacheArtifactFactory.register
class InductorTritonCacheArtifact(CacheArtifact):
    @staticmethod
    def type() -> str:
        return "inductor_triton"

    def populate_cache(self) -> None:
        if hashlib.sha256(self.content).hexdigest() != self.key:
            raise RuntimeError("Corrupt Inductor Triton runtime artifact")
        records = pickle.loads(self.content)
        if not isinstance(records, dict) or not all(
            isinstance(key, str) and isinstance(value, bytes)
            for key, value in records.items()
        ):
            raise RuntimeError("Invalid Inductor Triton runtime artifact")
        with _frozen_triton_kernels_lock:
            _frozen_triton_kernels.update(records)


@CacheArtifactFactory.register
class InductorCppCacheArtifact(CacheArtifact):
    @staticmethod
    def type() -> str:
        return "inductor_cpp"

    def populate_cache(self) -> None:
        if hashlib.sha256(self.content).hexdigest() != self.key:
            raise RuntimeError("Corrupt Inductor C++ runtime artifact")
        records = pickle.loads(self.content)
        if not isinstance(records, dict) or not all(
            isinstance(key, str) and isinstance(value, bytes)
            for key, value in records.items()
        ):
            raise RuntimeError("Invalid Inductor C++ runtime artifact")
        with _frozen_cpp_kernels_lock:
            _frozen_cpp_kernels.update(records)


def record_cpp_kernel(key: str, binary_path: str) -> None:
    with _capture_lock:
        owner = _capture
        if owner is None or owner.pid != os.getpid() or owner.sealed:
            return
        owner.cpp_kernels[key] = binary_path


def restore_cpp_kernel(key: str, binary_path: str) -> None:
    """Materialize a frozen C++ kernel binary at this host's cache path."""
    with _frozen_cpp_kernels_lock:
        payload = _frozen_cpp_kernels.get(key)
    if payload is None or os.path.exists(binary_path):
        return
    os.makedirs(os.path.dirname(binary_path), exist_ok=True)
    temporary = f"{binary_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(temporary, "wb") as output:
            output.write(payload)
        os.replace(temporary, binary_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _freeze_cpp_kernels(kernels: dict[str, str]) -> bytes:
    binaries: dict[str, bytes] = {}
    for key, binary_path in kernels.items():
        # Unforced synchronous loads never build; replay fails on them at load.
        if os.path.exists(binary_path):
            with open(binary_path, "rb") as binary:
                binaries[key] = binary.read()
    return pickle.dumps(binaries)


def record_triton_kernel(key: str, kernel: CachingAutotuner) -> None:
    with _capture_lock:
        owner = _capture
        if owner is None or owner.pid != os.getpid() or owner.sealed:
            return
        instances = owner.triton_kernels.setdefault(key, [])
        if not any(instance is kernel for instance in instances):
            instances.append(kernel)


def clear_triton_kernels() -> None:
    with _frozen_triton_kernels_lock:
        _frozen_triton_kernels.clear()


def load_triton_kernel(key: str) -> CachingAutotuner | None:
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult
    from torch._inductor.triton_bundler import StaticallyLaunchedAutotuner

    with _frozen_triton_kernels_lock:
        payload = _frozen_triton_kernels.get(key)
    if payload is None:
        return None
    record = pickle.loads(payload)
    if not isinstance(record, StaticallyLaunchedAutotuner) or record.cache_key != key:
        raise RuntimeError("Invalid Inductor Triton source identity")
    kernel = record.kernel
    if len(kernel.compile_results) != 1 or not isinstance(
        kernel.compile_results[0], StaticTritonCompileResult
    ):
        raise RuntimeError(
            "Inductor Triton runtime artifact has no selected static kernel"
        )
    result = kernel.compile_results[0]
    kernel.restore_after_unpickle(old_values=None)
    with kernel.lock:
        result.reload_cubin_path()
        # The artifact already selected its config. Rechecking local tuning caches
        # or running precompile can introduce new rblock/coordesc candidates.
        kernel._make_launchers()
    return kernel


class _UnresolvedTritonExport(RuntimeError):
    def __init__(self, sources: list[dict[str, object]]) -> None:
        self.sources = sources
        self.report = {"sources": sources}
        super().__init__(
            "Cannot export Triton sources: no fully resolved static launcher\n"
            + json.dumps(self.report, sort_keys=True)
        )


def _triton_export_rejection(kernel: CachingAutotuner) -> str | None:
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

    if len(kernel.launchers) != 1:
        return "launcher_count_not_one"
    result = getattr(kernel.launchers[0], "_compile_result", None)
    if not isinstance(result, StaticTritonCompileResult):
        return "non_static_compile_result"
    if kernel.inductor_meta.get("combo_tuning_groups") and not getattr(
        result.config, "found_by_combo_autotune", False
    ):
        return "combo_autotuning_incomplete"
    if (
        kernel.inductor_meta.get("coordinate_descent_tuning", False)
        and kernel._should_coordesc_tune
        and not getattr(result.config, "found_by_coordesc", False)
    ):
        return "coordinate_descent_tuning_incomplete"
    return None


def _triton_export_diagnostic(
    kernel: CachingAutotuner, reason: str
) -> dict[str, object]:
    compile_id = getattr(kernel, "compile_id", None)
    backward = getattr(kernel, "is_backward", None)
    configs = getattr(kernel, "configs", None)
    return {
        "reason": reason,
        "kernel_name": kernel.inductor_meta.get("kernel_name"),
        "filename": getattr(kernel, "filename", None),
        "compile_id": str(compile_id) if compile_id is not None else None,
        "direction": None
        if backward is None
        else "backward"
        if backward
        else "forward",
        "heuristic": getattr(getattr(kernel, "heuristic_type", None), "name", None),
        "config_count": len(configs) if configs is not None else None,
        "launcher_count": len(kernel.launchers),
        "compile_result_count": len(kernel.compile_results),
        "launcher_types": [type(launcher).__name__ for launcher in kernel.launchers],
        "launcher_result_types": [
            type(getattr(launcher, "_compile_result", None)).__name__
            for launcher in kernel.launchers
        ],
        "compile_result_types": [
            type(result).__name__ for result in kernel.compile_results
        ],
    }


def _freeze_triton_kernel(key: str, instances: list[CachingAutotuner]) -> bytes:
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult
    from torch._inductor.triton_bundler import StaticallyLaunchedAutotuner

    rejected: list[dict[str, object]] = []
    # Rendering can create an unexecuted instance of an already warmed source.
    for kernel in sorted(instances, key=lambda k: k._cached_launcher is None):
        with kernel.lock:
            reason = _triton_export_rejection(kernel)
            if reason is not None:
                rejected.append(_triton_export_diagnostic(kernel, reason))
                continue
            result = cast(
                StaticTritonCompileResult, kernel.launchers[0]._compile_result
            )
            old_results = kernel.compile_results
            old_cached_launcher = kernel._cached_launcher
            old_values = kernel.prepare_for_pickle()
            try:
                kernel.compile_results = [result]
                saved = copy.deepcopy(kernel)
                saved._reload_kernel = None
                cast(
                    StaticTritonCompileResult, saved.compile_results[0]
                ).reload_cubin_path()
                saved.prepare_for_caching()
                return pickle.dumps(
                    StaticallyLaunchedAutotuner(
                        key,
                        saved.inductor_meta.get("kernel_name", "unknown_kernel"),
                        saved,
                    )
                )
            finally:
                kernel.compile_results = old_results
                kernel.restore_after_unpickle(old_values)
                kernel._cached_launcher = old_cached_launcher
    raise _UnresolvedTritonExport(
        [{"source_key": key, "instance_count": len(instances), "instances": rejected}]
    )


class _RuntimeCapture:
    def __init__(self, cache_root: str | None) -> None:
        self.cache_root = cache_root
        self.pid = os.getpid()
        self.sealed = False
        self.policy = contextlib.ExitStack()
        self.finalize_lock = threading.Lock()
        self.triton_kernels: dict[str, list[CachingAutotuner]] = {}
        self.cpp_kernels: dict[str, str] = {}

    def seal(self) -> None:
        if self.sealed:
            raise RuntimeError(
                "The precompile runtime cache has already been finalized"
            )
        self.policy.enter_context(no_compilation())
        self.sealed = True


_capture: _RuntimeCapture | None = None
_capture_lock = threading.Lock()


@contextlib.contextmanager
def capture_runtime() -> Iterator[None]:
    """Own one producer lifetime, including strict validation and cleanup.

    When Triton can export its runtime cache, set an attempt-private
    TRITON_CACHE_DIR and TRITON_CACHE_AUTOTUNING=1 before application imports.
    Finalizing the cache seals this scope against further compiler work until the
    outer application cleanup has returned.
    """
    global _capture
    cache_root = None
    if (cache := _triton_runtime_cache()) is not None:
        from triton import knobs

        root = cache._runtime_cache_root(require_explicit=True)
        root.mkdir(parents=True, exist_ok=True)
        if (
            not knobs.autotuning.cache
            or os.environ.get("TRITON_CACHE_AUTOTUNING") != "1"
        ):
            raise RuntimeError(
                "Runtime capture requires TRITON_CACHE_AUTOTUNING=1 before "
                "application imports"
            )
        cache_root = str(root)
    if is_compilation_forbidden():
        raise RuntimeError(
            "Runtime capture cannot start while compilation is forbidden"
        )
    owner = _RuntimeCapture(cache_root)
    with _capture_lock:
        if _capture is not None:
            raise RuntimeError("Another precompile runtime capture is already active")
        _capture = owner
    try:
        yield
    finally:
        try:
            owner.policy.close()
        finally:
            with _capture_lock:
                owner.triton_kernels.clear()
                owner.cpp_kernels.clear()
                _capture = None


def finalize_runtime_cache(artifact: bytes | None) -> bytes:
    import torch
    from torch._inductor.async_compile import AsyncCompile

    cache = _triton_runtime_cache()

    owner = _capture
    if owner is None or owner.pid != os.getpid():
        raise RuntimeError(
            "precompile.finalize_cache requires the worker's capture_runtime scope"
        )
    with owner.finalize_lock:
        if owner.sealed:
            raise RuntimeError(
                "The precompile runtime cache has already been finalized"
            )
        if cache is not None and str(cache._runtime_cache_root()) != owner.cache_root:
            raise RuntimeError("The runtime-cache namespace changed during capture")
        artifacts = (
            CacheArtifactManager.deserialize(artifact) if artifact is not None else {}
        )
        if artifacts is None:
            raise RuntimeError("Cannot finalize an unreadable precompile cache")
        if any(
            kind in artifacts
            for kind in (
                TritonRuntimeCacheArtifact.type(),
                InductorTritonCacheArtifact.type(),
                InductorCppCacheArtifact.type(),
            )
        ):
            raise RuntimeError(
                "The precompile cache already contains frozen runtime dependencies"
            )
        AsyncCompile.drain_pending()
        if cache is not None:
            from triton.runtime._async_compile import wait_for_pending_compiles

            wait_for_pending_compiles()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        with _capture_lock:
            owner.seal()
            kernels = dict(owner.triton_kernels)
            cpp_kernels = dict(owner.cpp_kernels)
        static_kernels: dict[str, bytes] = {}
        rejected_sources: list[dict[str, object]] = []
        for key, instances in kernels.items():
            try:
                static_kernels[key] = _freeze_triton_kernel(key, instances)
            except _UnresolvedTritonExport as exc:
                rejected_sources.extend(exc.sources)
        if rejected_sources:
            error = _UnresolvedTritonExport(rejected_sources)
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "precompile_triton_export_failures",
                    "encoding": "json",
                },
                payload_fn=lambda: error.report,
                expect_trace_id=False,
                record_logging_overhead=False,
            )
            raise error
        static_payload = pickle.dumps(static_kernels)
        cpp_payload = _freeze_cpp_kernels(cpp_kernels)
        if cache is not None:
            payload = cache.export_runtime_cache(context=_runtime_context())
            artifacts[TritonRuntimeCacheArtifact.type()] = [
                TritonRuntimeCacheArtifact(hashlib.sha256(payload).hexdigest(), payload)
            ]
        artifacts[InductorTritonCacheArtifact.type()] = [
            InductorTritonCacheArtifact(
                hashlib.sha256(static_payload).hexdigest(), static_payload
            )
        ]
        artifacts[InductorCppCacheArtifact.type()] = [
            InductorCppCacheArtifact(
                hashlib.sha256(cpp_payload).hexdigest(), cpp_payload
            )
        ]
        serializer = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
        serializer.extend(artifacts.items())
        return serializer.to_bytes()


def prepare_runtime_cache(artifact: bytes | None) -> None:
    if artifact is None:
        artifacts = {}
    elif (artifacts := CacheArtifactManager.deserialize(artifact)) is None:
        raise RuntimeError(
            "Cannot prepare runtime dependencies from an unreadable precompile cache"
        )
    if len(artifacts.get(InductorTritonCacheArtifact.type(), ())) != 1:
        raise RuntimeError(
            "precompile.prepare_runtime requires a cache produced by "
            "precompile.finalize_cache"
        )
    runtime = artifacts.get(TritonRuntimeCacheArtifact.type(), ())
    if len(runtime) > 1:
        raise RuntimeError(
            "precompile.prepare_runtime found more than one Triton runtime cache"
        )
    for entry in runtime:
        entry.populate_cache()
