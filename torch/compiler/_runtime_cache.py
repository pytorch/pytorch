"""Runtime dependency transport for precompiled workers."""

from __future__ import annotations

import collections
import contextlib
import copy
import hashlib
import json
import logging
import os
import pickle
import threading
import types
from typing import cast, TYPE_CHECKING

from torch.compiler._cache import (
    _serialize_single_cache,
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
)
from torch.utils._appending_byte_serializer import AppendingByteSerializer


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torch._inductor.runtime.triton_heuristics import (
        CachingAutotuner,
        StaticTritonCompileResult,
    )


# Process-wide: once precompile.load installs them, a frozen kernel serves every
# later compile of the same Triton source in this process until fresh_cache() /
# clear_caches() (see FrozenTritonKernels).
_frozen_triton_kernels: dict[str, bytes] = {}
_loaded_triton_kernels: dict[str, CachingAutotuner] = {}
_frozen_triton_kernels_lock = threading.Lock()


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
            for key in records:
                _loaded_triton_kernels.pop(key, None)


def record_triton_kernel(key: str, kernel: CachingAutotuner) -> None:
    with _capture_lock:
        owner = _capture
        if owner is None or owner.pid != os.getpid():
            return
        instances = owner.triton_kernels.setdefault(key, [])
        if not any(instance is kernel for instance in instances):
            instances.append(kernel)


def clear_triton_kernels() -> None:
    with _frozen_triton_kernels_lock:
        _frozen_triton_kernels.clear()
        _loaded_triton_kernels.clear()


def load_triton_kernel(key: str) -> CachingAutotuner | None:
    with _frozen_triton_kernels_lock:
        if (kernel := _loaded_triton_kernels.get(key)) is not None:
            return kernel
        payload = _frozen_triton_kernels.get(key)
        if payload is None:
            return None
        kernel = _load_frozen_triton_kernel(key, payload)
        _loaded_triton_kernels[key] = kernel
        return kernel


def _load_frozen_triton_kernel(key: str, payload: bytes) -> CachingAutotuner:
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult
    from torch._inductor.triton_bundler import StaticallyLaunchedAutotuner

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
            "Cannot export Triton sources: no fully resolved static launcher. Run "
            "every Triton kernel the serving process needs inside the capture block.\n"
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
    # CachingAutotuner.precompile drops configs once it has compile_results.
    configs = getattr(kernel, "configs", None)
    return {
        "reason": reason,
        "kernel_name": kernel.inductor_meta.get("kernel_name"),
        "filename": getattr(kernel, "filename", None),
        "compile_id": str(compile_id) if compile_id is not None else None,
        "direction": None
        if compile_id is None
        else ("backward" if kernel.is_backward else "forward"),
        "heuristic": getattr(getattr(kernel, "heuristic_type", None), "name", None),
        "config_count": len(kernel.compile_results if configs is None else configs),
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


def _detached_copy(fn: object) -> object:
    # A shallow copy's bound methods (and cache factories) still point at the live
    # object, so pickling the copy would pickle the live object too.
    saved = copy.copy(fn)
    for name, value in vars(fn).items():
        if getattr(value, "__self__", None) is fn:
            setattr(saved, name, types.MethodType(value.__func__, saved))
        elif (
            isinstance(value, collections.defaultdict)
            and isinstance(factory := value.default_factory, types.MethodType)
            and factory.__self__ is fn
        ):
            rebound = types.MethodType(factory.__func__, saved)
            setattr(saved, name, collections.defaultdict(rebound, value))
    return saved


def _retain_cubin(result: StaticTritonCompileResult) -> None:
    # A launched kernel has dropped its cubin bytes, and pickling drops its cubin
    # path, so the frozen record must carry the bytes itself.
    static_kernel = result.kernel
    if static_kernel.cubin_raw is None:
        result.reload_cubin_path()
        with open(static_kernel.cubin_path, "rb") as f:
            static_kernel.cubin_raw = f.read()


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
            # run() reads the live kernel's launchers without taking its lock, so
            # freeze a copy and leave the live kernel untouched.
            saved = type(kernel).__new__(type(kernel))
            saved.__dict__.update(kernel.__dict__)
            saved.fn = _detached_copy(kernel.fn)
            saved_result = copy.copy(result)
            saved_result.kernel = copy.copy(result.kernel)
        saved.prepare_for_pickle()
        saved.compile_results = [saved_result]
        saved._reload_kernel = None
        try:
            _retain_cubin(saved_result)
        except (OSError, RuntimeError) as exc:
            with kernel.lock:
                diagnostic = _triton_export_diagnostic(kernel, "cubin_unavailable")
            diagnostic["error"] = str(exc)
            rejected.append(diagnostic)
            continue
        return pickle.dumps(
            StaticallyLaunchedAutotuner(
                key, saved.inductor_meta.get("kernel_name", "unknown_kernel"), saved
            )
        )
    kernel_name = instances[0].inductor_meta.get("kernel_name")
    raise _UnresolvedTritonExport(
        [
            {
                "source_key": key,
                "kernel_name": kernel_name,
                "instance_count": len(instances),
                "instances": rejected,
            }
        ]
    )


class _RuntimeCapture:
    def __init__(self) -> None:
        self.pid = os.getpid()
        # The precompile.capture blocks open in this process. Overlapping
        # captures are unsupported; counting them only keeps a misuse from
        # closing the scope under the other capture.
        self.captures = 0
        self.freeze_lock = threading.Lock()
        self.triton_kernels: dict[str, list[CachingAutotuner]] = {}


_capture: _RuntimeCapture | None = None
_capture_lock = threading.Lock()


def _active_capture() -> _RuntimeCapture | None:
    # A forked child inherits the parent's scope object; it is not the child's.
    owner = _capture
    return owner if owner is not None and owner.pid == os.getpid() else None


@contextlib.contextmanager
def capture_scope() -> Iterator[None]:
    """The scope a precompile.capture block records into, closed when the
    block exits. Overlapping captures are unsupported; a misused overlap shares
    the scope rather than closing it under the other capture.
    """
    global _capture
    with _capture_lock:
        owner = _active_capture()
        if owner is None:
            owner = _capture = _RuntimeCapture()
        owner.captures += 1
    try:
        yield
    finally:
        with _capture_lock:
            owner.captures -= 1
            last = owner.captures == 0
            if last:
                owner.triton_kernels.clear()
                _capture = None


def _freeze_triton_kernels(kernels: dict[str, list[CachingAutotuner]]) -> bytes:
    from torch._logging import trace_structured

    static_kernels: dict[str, bytes] = {}
    rejected_sources: list[dict[str, object]] = []
    for key, instances in kernels.items():
        try:
            static_kernels[key] = _freeze_triton_kernel(key, instances)
        except _UnresolvedTritonExport as exc:
            rejected_sources.extend(exc.sources)
    if rejected_sources:
        error = _UnresolvedTritonExport(rejected_sources)
        trace_structured(
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
    return pickle.dumps(static_kernels)


def freeze_runtime_cache(
    snapshot: Callable[[], bytes | None], write: Callable[[bytes | None], None]
) -> None:
    """Freeze the scope's static Triton kernels into the cache artifact and pass
    it to ``write``.

    Pending async compiles are waited on before ``snapshot`` renders the
    artifact, so what they record is in it. A kernel that cannot be frozen
    raises, since precompile.load could not serve it without compiling. The
    caller must stop compiling before calling this: a kernel submitted on
    another thread after the wait is neither waited on nor frozen.
    """
    import torch
    from torch._inductor.async_compile import AsyncCompile

    owner = _active_capture()
    if owner is None:
        raise RuntimeError("precompile.capture has no runtime capture scope")
    with owner.freeze_lock:
        # Keep the workers: a save() checkpoint is followed by more of the block.
        AsyncCompile.drain_pending(shutdown=False)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        artifact = snapshot()
        artifacts = (
            CacheArtifactManager.deserialize(artifact) if artifact is not None else {}
        )
        if artifacts is None:
            raise RuntimeError("Cannot freeze an unreadable precompile cache")
        if InductorTritonCacheArtifact.type() in artifacts:
            raise RuntimeError(
                "The precompile cache already contains frozen runtime dependencies"
            )
        with _capture_lock:
            kernels = {key: list(found) for key, found in owner.triton_kernels.items()}
        static_payload = _freeze_triton_kernels(kernels)
        artifacts[InductorTritonCacheArtifact.type()] = [
            InductorTritonCacheArtifact(
                hashlib.sha256(static_payload).hexdigest(), static_payload
            )
        ]
        serializer = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
        serializer.extend(artifacts.items())
        write(serializer.to_bytes())
