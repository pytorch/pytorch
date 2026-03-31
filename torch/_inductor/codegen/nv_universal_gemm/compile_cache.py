# mypy: allow-untyped-defs
"""
Persistent disk cache for NVIDIA Universal GEMM compiled artifacts.

Compiled CuTeDSL kernels are exported as shared libraries (.so) via export_to_c.
On subsequent runs the .so is loaded via cute.runtime.load_module (~1ms) instead
of re-compiling via MLIR (~100ms+ per kernel).

cute.compile() is not thread-safe (MLIR thread-local state), so precompilation
spawns a subprocess per kernel (like CUTLASS uses subprocess for nvcc). The
ThreadPoolExecutor in make_precompile_fn handles parallelism.

Reuses existing inductor caching infrastructure:
  - torch._inductor.codecache.{cache_dir, code_hash, get_lock_dir, LOCK_TIMEOUT}
  - torch.utils._filelock.FileLock

Controls:
  TORCHINDUCTOR_NVGEMM_CACHE_ENABLED=0  - disable persistent .so cache (default: enabled)
"""

import ctypes
import json
import logging
import os
import subprocess
import sys
from distutils.ccompiler import CCompiler, new_compiler
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


log = logging.getLogger(__name__)

COMPILE_ONLY: bool = False
CACHE_ENABLED: bool = os.environ.get("TORCHINDUCTOR_NVGEMM_CACHE_ENABLED", "1") == "1"

EXPORT_FUNC_NAME = "func"

_compiler: CCompiler | None = None
_runtime_libs_loaded = False


def _get_compiler() -> CCompiler:
    global _compiler
    if _compiler is None:
        _compiler = new_compiler()
    return _compiler


def _ensure_runtime_libs():
    global _runtime_libs_loaded
    if _runtime_libs_loaded:
        return
    import cutlass.cute as cute

    for path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
        if Path(path).exists():
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    _runtime_libs_loaded = True


@lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    from torch._inductor.codecache import code_hash

    try:
        from importlib.metadata import version

        extra = f"cutlass-dsl={version('nvidia-cutlass-dsl')}"
    except Exception:
        extra = "cutlass-dsl=unknown"
    return code_hash(f"nvgemm-{torch.__version__}", extra)


def _get_cache_dir() -> Path:
    from torch._inductor.codecache import cache_dir

    fp = _compute_source_fingerprint()
    cache_path = Path(cache_dir()) / "nvgemm_cache" / fp[:3]
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _compute_cache_key(
    kernel_name: str,
    input_tensors: tuple[torch.Tensor, ...],
    out_tensor: torch.Tensor,
    accumulator_type: str,
) -> str:
    from torch._inductor.codecache import code_hash

    all_tensors = list(input_tensors) + [out_tensor]
    key_str = repr((
        kernel_name,
        [(tuple(t.shape), str(t.dtype), tuple(t.stride())) for t in all_tensors],
        accumulator_type,
        _compute_source_fingerprint(),
    ))
    return code_hash(key_str)


def _extract_jit_fn(compiled_obj: Any) -> Any:
    if hasattr(compiled_obj, "__closure__") and compiled_obj.__closure__:
        for cell in compiled_obj.__closure__:
            try:
                val = cell.cell_contents
                if hasattr(val, "export_to_c"):
                    return val
            except ValueError:
                continue
    return None


def _export_to_so(jit_fn: Any, so_path: Path) -> None:
    so_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path = so_path.with_suffix(".o")
    jit_fn.export_to_c(
        object_file_path=str(obj_path),
        function_name=EXPORT_FUNC_NAME,
    )
    _get_compiler().link_shared_object([str(obj_path)], str(so_path))
    if obj_path.exists():
        obj_path.unlink()


def _load_from_so(so_path: Path) -> Any:
    import cutlass.cute as cute

    _ensure_runtime_libs()
    m = cute.runtime.load_module(str(so_path), enable_tvm_ffi=True)
    return m[EXPORT_FUNC_NAME]


def _compile_with_cached_binary(kernel: Any, args: Any, so_path: Path) -> Any:
    """Compile kernel using a cached .so binary.

    Patches cute.compile to return the pre-loaded binary for the main kernel,
    while letting other internal cute.compile calls go through normally.
    """
    import cutlass.cute as cute

    loaded_fn = _load_from_so(so_path)
    original_compile = cute.compile
    target_fn = getattr(kernel, "impl", None)

    class _CachedCompile:
        def __init__(self, original):
            self._original = original

        def __call__(self, fn, *compile_args, **kwargs):
            if target_fn is not None and fn is target_fn:
                return loaded_fn
            return self._original(fn, *compile_args, **kwargs)

        def __getitem__(self, options):
            return _CachedCompileWithOptions(self._original, options)

        def __getattr__(self, name):
            return getattr(self._original, name)

    class _CachedCompileWithOptions:
        def __init__(self, original, options):
            self._original = original
            self._options = options

        def __call__(self, fn, *compile_args, **kwargs):
            if target_fn is not None and fn is target_fn:
                return loaded_fn
            return self._original[self._options](fn, *compile_args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._original[self._options], name)

    patched = _CachedCompile(original_compile)
    cute.compile = patched
    try:
        return kernel.compile(args)
    finally:
        cute.compile = original_compile


def nvgemm_compile_and_cache(
    kernel: Any,
    args: Any,
    kernel_name: str | None = None,
    input_tensors: tuple[torch.Tensor, ...] | None = None,
    out_tensor: torch.Tensor | None = None,
) -> Any:
    """Compile a cutlass_api kernel with persistent disk caching.

    Follows the same pattern as CUDACodeCache.compile(): hash-based file
    caching with FileLock for concurrent safety. Returns a CompiledArtifact
    on success, or None in COMPILE_ONLY mode.
    """
    if kernel_name is None:
        kernel_name = kernel.metadata.kernel_name

    if not CACHE_ENABLED or input_tensors is None or out_tensor is None:
        artifact = kernel.compile(args)
        return None if COMPILE_ONLY else artifact

    cache_key = _compute_cache_key(
        kernel_name, input_tensors, out_tensor,
        str(getattr(args, "accumulator_type", "unknown")),
    )
    cache_dir = _get_cache_dir()
    so_path = cache_dir / f"{cache_key}.so"

    from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT
    from torch.utils._filelock import FileLock

    lock_path = os.path.join(get_lock_dir(), f"nvgemm_{cache_key}.lock")

    with FileLock(lock_path, timeout=LOCK_TIMEOUT):
        if so_path.exists():
            if COMPILE_ONLY:
                return None
            return _compile_with_cached_binary(kernel, args, so_path)

        artifact = kernel.compile(args)

        jit_fn = _extract_jit_fn(artifact.compiled_obj)
        if jit_fn is not None:
            try:
                _export_to_so(jit_fn, so_path)
                log.debug("Cached NVGEMM kernel %s -> %s", kernel_name, so_path)
            except Exception as e:
                log.debug("Failed to cache NVGEMM kernel %s: %s", kernel_name, e)

        return None if COMPILE_ONLY else artifact


def precompile_in_subprocess(
    kernel_name: str,
    input_tensor_metas: list[dict[str, Any]],
    output_tensor_meta: dict[str, Any],
    accumulator_type: str,
    variant: str,
    scale_info: dict[str, str] | None = None,
) -> None:
    """Compile a single NVGEMM kernel in a subprocess to populate the disk cache.

    Like CUDACodeCache.compile() uses subprocess.check_output(nvcc ...), this
    spawns a subprocess for CuTeDSL compilation. Thread-safe because each call
    gets its own process.
    """
    task_json = json.dumps({
        "kernel_name": kernel_name,
        "input_tensor_metas": input_tensor_metas,
        "output_tensor_meta": output_tensor_meta,
        "accumulator_type": accumulator_type,
        "variant": variant,
        "scale_info": scale_info,
    })

    try:
        subprocess.check_output(
            [
                sys.executable, "-m",
                "torch._inductor.codegen.nv_universal_gemm.compile_worker",
                "--task", task_json,
            ],
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    except subprocess.CalledProcessError as e:
        log.debug(
            "NVGEMM subprocess compilation failed for %s: %s",
            kernel_name,
            e.output.decode("utf-8", errors="replace") if e.output else str(e),
        )
    except subprocess.TimeoutExpired:
        log.debug("NVGEMM subprocess compilation timed out for %s", kernel_name)


def clear_cache() -> None:
    import shutil

    cache_dir = _get_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        log.info("Cleared NVGEMM compile cache at %s", cache_dir)
