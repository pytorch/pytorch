"""Runtime dependency transport for precompiled workers."""

from __future__ import annotations

import atexit
import collections
import contextlib
import copy
import gc
import hashlib
import json
import logging
import os
import pickle
import shutil
import tempfile
import threading
import types
from pathlib import Path
from typing import cast, TYPE_CHECKING

from torch.compiler._cache import (
    _serialize_single_cache,
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
)
from torch.compiler._no_compile import is_compilation_forbidden
from torch.utils._appending_byte_serializer import AppendingByteSerializer


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

    from triton.runtime.autotuner import Autotuner

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
# Keyed like CppCodeCache: by source and full build command, which includes the
# producer's absolute include and library paths.
_frozen_cpp_kernels: dict[str, bytes] = {}
_frozen_cpp_kernels_lock = threading.Lock()


def _triton_runtime_cache() -> ModuleType | None:
    """Return the Triton runtime-cache transport, or None without a usable Triton.

    Without it, a capture still freezes Inductor's static Triton launchers and
    C++ kernels, but kernels launched directly through Triton's JIT are not captured
    and compile again on first use in the serving process.
    """
    try:
        from triton import knobs
    except ImportError:
        return None
    if not hasattr(knobs, "autotuning") or not hasattr(knobs.cache, "manager_class"):
        return None
    from torch.compiler import _triton_runtime_cache

    return _triton_runtime_cache


def _private_directory(prefix: str) -> str:
    # Kept until the process exits: kernels compiled while Triton used it keep
    # loading their binaries from it.
    directory = tempfile.mkdtemp(prefix=prefix)
    atexit.register(lambda: shutil.rmtree(directory, ignore_errors=True))
    return directory


def _live_autotuners() -> list[Autotuner]:
    from triton.runtime.autotuner import Autotuner

    # type() rather than isinstance(): some module objects warn on __class__.
    return [obj for obj in gc.get_objects() if issubclass(type(obj), Autotuner)]


def _importable(root: Path, digest: str) -> bool:
    from torch.compiler._triton_runtime_cache import _READY

    if not root.is_dir():
        return not root.exists()
    marker = root / _READY
    return not any(root.iterdir()) or (
        marker.is_file() and marker.read_text() == digest
    )


def _serving_triton_root(cache: ModuleType, digest: str) -> str:
    """Point Triton at a cache directory to import a runtime cache into, and turn
    its autotuning cache on, for the rest of the process.

    That is TRITON_CACHE_DIR if it is empty or already holds this cache, and a
    fresh directory otherwise, since the import refuses to mix with unrelated
    entries (Inductor points TRITON_CACHE_DIR at its shared cache once it
    compiles). Autotuners fix cache_results when they are decorated, so
    existing ones are switched too, or they would benchmark again instead of
    reading the imported decisions.
    """
    from triton import knobs

    configured = os.environ.get("TRITON_CACHE_DIR")
    if configured is None or not _importable(Path(configured), digest):
        os.environ["TRITON_CACHE_DIR"] = _private_directory(
            "torch_precompile_triton_load_"
        )
    # Through the environment rather than a knobs.cache.dir override, which would
    # outlive any later TRITON_CACHE_DIR.
    del knobs.cache.dir
    knobs.autotuning.cache = True
    for tuner in _live_autotuners():
        tuner.cache_results = not knobs.runtime.interpret
    return str(cache.runtime_cache_root())


def _runtime_context() -> dict[str, str | None]:
    import torch
    from torch._inductor.codecache import torch_key

    return {"torch_key": torch_key().hex(), "torch_cuda": torch.version.cuda}


# (artifact key, runtime-cache root) pairs already imported by this process, so
# a second precompile.load does not extract the same cache again.
_imported_triton_runtime: set[tuple[str, str]] = set()
_imported_triton_runtime_lock = threading.Lock()


@CacheArtifactFactory.register
class TritonRuntimeCacheArtifact(CacheArtifact):
    @staticmethod
    def type() -> str:
        return "triton_runtime"

    @staticmethod
    def populate_first() -> bool:
        # Graph artifacts can launch Triton kernels as they load.
        return True

    def populate_cache(self) -> None:
        self.import_into_triton(strict=is_compilation_forbidden())

    def import_into_triton(self, *, strict: bool) -> None:
        cache = _triton_runtime_cache()
        problem = cause = root = None
        if cache is None:
            problem = "Triton is not installed or too old to import one"
        elif _active_capture() is not None:
            problem = "it cannot be imported inside a precompile.capture block"
        else:
            # A failing lookup is reported by _serving_triton_root below.
            with contextlib.suppress(Exception), _imported_triton_runtime_lock:
                served = (self.key, str(cache.runtime_cache_root()))
                if served in _imported_triton_runtime:
                    return
            try:
                root = _serving_triton_root(cache, self.key)
            except Exception as exc:
                problem = f"Triton has no cache directory to import it into ({exc})"
                cause = exc
        if cache is None or root is None:
            message = (
                f"The precompile cache contains a Triton runtime cache, but {problem}"
            )
            if strict:
                raise RuntimeError(message) from cause
            log.warning(
                "%s; Triton JIT kernels will compile or autotune on first use", message
            )
            return
        imported = (self.key, root)
        with _imported_triton_runtime_lock:
            if imported in _imported_triton_runtime:
                return
            try:
                if hashlib.sha256(self.content).hexdigest() != self.key:
                    raise RuntimeError("Corrupt Triton runtime-cache artifact")
                cache.import_runtime_cache(self.content, context=_runtime_context())
            except Exception as exc:
                if strict:
                    raise
                # Raising here would stop the artifacts populated after this one
                # (populate_first), leaving the frozen Inductor kernels cold too.
                # The TRITON_CACHE_DIR and autotuning settings _serving_triton_root
                # made persist, so later JIT compiles land in that directory.
                log.warning(
                    "Could not import the precompile cache's Triton runtime cache "
                    "(%s); Triton JIT kernels will compile or autotune on first use",
                    exc,
                )
                return
            _imported_triton_runtime.add(imported)


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
        if owner is None or owner.pid != os.getpid():
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
        if owner is None or owner.pid != os.getpid():
            return
        instances = owner.triton_kernels.setdefault(key, [])
        if not any(instance is kernel for instance in instances):
            instances.append(kernel)


def record_inductor_triton_binary(cache_key: str) -> None:
    # Inductor's binaries, including losing autotune candidates, share the
    # Triton cache with JIT kernels; inductor_triton already ships the ones
    # replay needs, so triton_runtime excludes these keys.
    with _capture_lock:
        owner = _capture
        if owner is not None and owner.pid == os.getpid():
            owner.inductor_triton_keys.add(cache_key)


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
        # Restores Triton's cache settings when the scope closes.
        self.triton_settings = contextlib.ExitStack()
        # The fresh Triton cache directory whose JIT cache the scope ships.
        self.cache_root = _own_triton_cache(self.triton_settings)
        self.freeze_lock = threading.Lock()
        self.triton_kernels: dict[str, list[CachingAutotuner]] = {}
        self.cpp_kernels: dict[str, tuple[str, Callable[[], object]]] = {}
        self.inductor_triton_keys: set[str] = set()


_capture: _RuntimeCapture | None = None
_capture_lock = threading.Lock()


def _active_capture() -> _RuntimeCapture | None:
    # A forked child inherits the parent's scope object; it is not the child's.
    owner = _capture
    return owner if owner is not None and owner.pid == os.getpid() else None


def _own_triton_cache(settings: contextlib.ExitStack) -> str | None:
    """Point Triton at a fresh cache directory and turn its autotuning cache on
    until ``settings`` closes, so the directory holds exactly the Triton JIT
    kernels and autotuning decisions the scope used.

    Autotuners fix cache_results when they are decorated, so existing ones are
    switched on for the scope too.
    """
    if _triton_runtime_cache() is None:
        return None
    from triton import knobs
    from triton.runtime import driver

    root = _private_directory("torch_precompile_triton_")
    # Triton builds its driver's utility module (cuda_utils, hip_utils) into
    # the cache directory and loads it once per process, so a process that
    # already loaded it never builds it into the new directory. Copy the loaded
    # build so the export carries it. _active, not active: that creates a driver.
    utils = getattr(driver._active or driver._default, "utils", None)
    loaded = getattr(getattr(utils, "load_binary", None), "__self__", None)
    # The module's directory may since have been deleted; a serving host then
    # rebuilds it, as it does when this process never loaded one.
    module = getattr(loaded, "__file__", None)
    if module is not None and (module := Path(module)).is_file():
        entry = Path(root) / module.parent.name
        entry.mkdir(exist_ok=True)
        shutil.copy2(module, entry / module.name)
    settings.enter_context(knobs.cache.scope())
    settings.enter_context(knobs.autotuning.scope())
    knobs.cache.dir = root
    knobs.autotuning.cache = True
    tuners = _live_autotuners()
    restore = [(tuner, tuner.cache_results) for tuner in tuners]
    for tuner in tuners:
        tuner.cache_results = not knobs.runtime.interpret

    def restore_cache_results() -> None:
        for tuner, cache_results in restore:
            tuner.cache_results = cache_results

    settings.callback(restore_cache_results)
    return root


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
                owner.cpp_kernels.clear()
                owner.inductor_triton_keys.clear()
                _capture = None
        if last:
            owner.triton_settings.close()


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


def _current_root(cache: ModuleType) -> str | None:
    try:
        return str(cache.runtime_cache_root())
    except Exception:
        return None


def freeze_runtime_cache(
    snapshot: Callable[[], bytes | None], write: Callable[[bytes | None], None]
) -> None:
    """Freeze the scope's static Triton kernels, C++ kernel binaries and Triton
    JIT cache into the cache artifact and pass it to ``write``.

    Pending async compiles are waited on and every recorded C++ kernel is built
    before ``snapshot`` renders the artifact, so what they record is in it. A
    kernel that cannot be frozen raises, since precompile.load could not serve
    it without compiling. The caller must stop compiling before calling this: a
    kernel submitted on another thread after the wait is neither waited on nor
    frozen.
    """
    import torch
    from torch._inductor.async_compile import AsyncCompile

    owner = _active_capture()
    if owner is None:
        raise RuntimeError("precompile.capture has no runtime capture scope")
    cache = _triton_runtime_cache() if owner.cache_root is not None else None
    with owner.freeze_lock:
        if cache is not None and _current_root(cache) != owner.cache_root:
            raise RuntimeError("Triton's cache directory changed during the capture")
        # Keep the workers: a save() checkpoint is followed by more of the block.
        AsyncCompile.drain_pending(shutdown=False)
        with _capture_lock:
            cpp_loads = [load for _, load in owner.cpp_kernels.values()]
        # Finishes pending C++ builds and builds loads that were never forced, so
        # no binary is missing or partially written when it is frozen.
        for load in cpp_loads:
            load()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        artifact = snapshot()
        artifacts = (
            CacheArtifactManager.deserialize(artifact) if artifact is not None else {}
        )
        if artifacts is None:
            raise RuntimeError("Cannot freeze an unreadable precompile cache")
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
        with _capture_lock:
            kernels = {key: list(found) for key, found in owner.triton_kernels.items()}
            cpp_kernels = {key: path for key, (path, _) in owner.cpp_kernels.items()}
            inductor_triton_keys = frozenset(owner.inductor_triton_keys)
        static_payload = _freeze_triton_kernels(kernels)
        cpp_payload = _freeze_cpp_kernels(cpp_kernels)
        if cache is not None:
            payload = cache.export_runtime_cache(
                context=_runtime_context(), exclude=inductor_triton_keys
            )
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
        write(serializer.to_bytes())
