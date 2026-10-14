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
from torch.compiler._no_compile import is_compilation_forbidden, no_compilation
from torch.utils._appending_byte_serializer import AppendingByteSerializer


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torch._inductor.runtime.triton_heuristics import (
        CachingAutotuner,
        StaticTritonCompileResult,
    )


def _precompile_error(message: str) -> Exception:
    from torch._precompile import PrecompileError

    return PrecompileError(message)


# Process-wide: once precompile.load installs them, a frozen kernel serves every
# later compile of the same Triton source in this process until fresh_cache() /
# clear_caches() (see FrozenTritonKernels).
_frozen_triton_kernels: dict[str, bytes] = {}
_loaded_triton_kernels: dict[str, CachingAutotuner] = {}
_frozen_triton_kernels_lock = threading.Lock()
# Keyed like CppCodeCache: by source and full build command, which includes the
# producer's absolute include and library paths.
_frozen_cpp_kernels: dict[str, bytes] = {}
_frozen_cpp_kernels_lock = threading.Lock()


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


def clear_cpp_kernels() -> None:
    with _frozen_cpp_kernels_lock:
        _frozen_cpp_kernels.clear()


def record_cpp_kernel(key: str, binary_path: str, load: Callable[[], object]) -> None:
    with _capture_lock:
        owner = _capture
        if owner is None or owner.pid != os.getpid() or owner.sealed:
            return
        owner.cpp_kernels[key] = (binary_path, load)


def has_frozen_cpp_kernel(key: str) -> bool:
    with _frozen_cpp_kernels_lock:
        return key in _frozen_cpp_kernels


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
    missing = sorted(key for key, path in kernels.items() if not os.path.exists(path))
    if missing:
        raise RuntimeError(f"Cannot export C++ kernels with no binary: {missing}")
    binaries: dict[str, bytes] = {}
    for key, binary_path in kernels.items():
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
            diagnostic = _triton_export_diagnostic(kernel, "cubin_unavailable")
            diagnostic["error"] = str(exc)
            rejected.append(diagnostic)
            continue
        return pickle.dumps(
            StaticallyLaunchedAutotuner(
                key, saved.inductor_meta.get("kernel_name", "unknown_kernel"), saved
            )
        )
    raise _UnresolvedTritonExport(
        [{"source_key": key, "instance_count": len(instances), "instances": rejected}]
    )


class _RuntimeCapture:
    def __init__(self, *, explicit: bool) -> None:
        self.pid = os.getpid()
        # Opened by capture_runtime(), else by the first open precompile.capture
        # and closed by the last one to exit.
        self.explicit = explicit
        self.captures = 0
        self.sealed = False
        self.policy = contextlib.ExitStack()
        self.freeze_lock = threading.Lock()
        self.triton_kernels: dict[str, list[CachingAutotuner]] = {}
        self.cpp_kernels: dict[str, tuple[str, Callable[[], object]]] = {}

    def seal(self) -> None:
        self.policy.enter_context(no_compilation())
        self.sealed = True

    def unseal(self) -> None:
        self.policy.close()
        self.policy = contextlib.ExitStack()
        self.sealed = False


_capture: _RuntimeCapture | None = None
_capture_lock = threading.Lock()


def _active_capture() -> _RuntimeCapture | None:
    # A forked child inherits the parent's scope object; it is not the child's.
    owner = _capture
    return owner if owner is not None and owner.pid == os.getpid() else None


def _close(owner: _RuntimeCapture) -> None:
    global _capture
    try:
        owner.policy.close()
    finally:
        with _capture_lock:
            owner.triton_kernels.clear()
            owner.cpp_kernels.clear()
            if _capture is owner:
                _capture = None


@contextlib.contextmanager
def capture_runtime() -> Iterator[None]:
    """Own one producer lifetime, including strict validation and cleanup.

    The capture inside it seals the scope against further compiler work until
    the outer application cleanup has returned.
    """
    global _capture
    if is_compilation_forbidden():
        raise _precompile_error(
            "precompile.capture_runtime cannot start while compilation is forbidden"
        )
    owner = _RuntimeCapture(explicit=True)
    with _capture_lock:
        if _active_capture() is not None:
            raise _precompile_error(
                "Another precompile runtime capture or an open precompile.capture "
                "is already active"
            )
        _capture = owner
    try:
        yield
    finally:
        _close(owner)


@contextlib.contextmanager
def capture_scope() -> Iterator[None]:
    """The scope a precompile.capture block records into: the enclosing
    capture_runtime() scope, else one shared by the captures open in this process.
    """
    global _capture
    with _capture_lock:
        owner = _active_capture()
        if owner is None:
            owner = _capture = _RuntimeCapture(explicit=False)
        owner.captures += 1
    try:
        yield
    finally:
        with _capture_lock:
            owner.captures -= 1
            last = not owner.explicit and owner.captures == 0
        if last:
            _close(owner)


def seal_runtime_scope() -> None:
    """Seal an enclosing capture_runtime() scope whose cache is already written."""
    owner = _active_capture()
    if owner is None or not owner.explicit:
        return
    with owner.freeze_lock, _capture_lock:
        if not owner.sealed:
            owner.seal()


def _freeze_triton_kernels(
    kernels: dict[str, list[CachingAutotuner]], *, strict: bool
) -> bytes:
    import torch

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
        if strict:
            raise error
        log.warning(
            "precompile left %d Triton kernel source(s) out of the cache, which "
            "compile again on first use in the serving process: %s",
            len(rejected_sources),
            error,
        )
    return pickle.dumps(static_kernels)


def freeze_runtime_cache(
    artifact: bytes | None,
    write: Callable[[bytes | None], None],
    *,
    final: bool,
    compiled: bool,
) -> None:
    """Freeze the scope's static Triton kernels and C++ kernel binaries into the
    cache artifact and pass it to ``write``.

    Pending async compiles are waited on and every recorded C++ kernel is built
    first. A ``final`` write inside a
    capture_runtime() scope then seals it so any later compile raises before
    kernels are frozen, and keeps it sealed unless freezing, serialization or
    ``write`` raises. A capture_runtime() scope also refuses a kernel that cannot
    be frozen, where a plain capture leaves it out with a warning. A capture
    that did not run eagerly (``compiled``) but saved no artifact has no
    dependencies to carry; a capture_runtime() scope refuses it, since strict
    precompile.load could not serve it either. The caller must stop compiling
    before calling this: a kernel submitted on another thread after the wait but
    before the seal is neither waited on nor rejected, and is left out of the
    cache.
    """
    import torch
    from torch._inductor.async_compile import AsyncCompile

    owner = _active_capture()
    if owner is None:
        raise RuntimeError("precompile.capture has no runtime capture scope")
    if compiled and artifact is None:
        if owner.explicit:
            raise RuntimeError(
                "precompile.capture_runtime requires the compiled cache artifact of "
                "a compiled capture, and this capture saved none"
            )
        write(None)
        return
    with owner.freeze_lock:
        if owner.sealed:
            raise RuntimeError(
                "The precompile.capture_runtime scope was already sealed by an "
                "earlier capture"
            )
        artifacts = (
            CacheArtifactManager.deserialize(artifact) if artifact is not None else {}
        )
        if artifacts is None:
            raise RuntimeError("Cannot freeze an unreadable precompile cache")
        if any(
            kind in artifacts
            for kind in (
                InductorTritonCacheArtifact.type(),
                InductorCppCacheArtifact.type(),
            )
        ):
            raise RuntimeError(
                "The precompile cache already contains frozen runtime dependencies"
            )
        # Drain before sealing: compile result callbacks re-check
        # check_compilation_allowed, so draining under the seal rejects them.
        AsyncCompile.drain_pending()
        with _capture_lock:
            cpp_loads = [load for _, load in owner.cpp_kernels.values()]
        # Finishes pending C++ builds and builds loads that were never forced, so
        # no binary is missing or partially written when it is frozen.
        for load in cpp_loads:
            load()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        seal = final and owner.explicit
        with _capture_lock:
            if seal:
                owner.seal()
            kernels = {key: list(found) for key, found in owner.triton_kernels.items()}
            cpp_kernels = {key: path for key, (path, _) in owner.cpp_kernels.items()}
        try:
            static_payload = _freeze_triton_kernels(kernels, strict=owner.explicit)
            cpp_payload = _freeze_cpp_kernels(cpp_kernels)
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
            write(serializer.to_bytes())
        except BaseException:
            if seal:
                with _capture_lock:
                    owner.unseal()
            raise


def prepare_runtime_cache(artifact: bytes | None) -> None:
    if artifact is None:
        artifacts = {}
    elif (artifacts := CacheArtifactManager.deserialize(artifact)) is None:
        raise RuntimeError(
            "Cannot prepare runtime dependencies from an unreadable precompile cache"
        )
    if len(artifacts.get(InductorTritonCacheArtifact.type(), ())) != 1:
        raise RuntimeError(
            "precompile.prepare_runtime requires a cache whose capture froze its "
            "runtime dependencies"
        )
