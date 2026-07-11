"""
System ptxas replacement for CUTLASS DSL.
Environment variables:
    CUTE_DSL_PTXAS_PATH    - Path to ptxas (e.g., /usr/local/cuda/bin/ptxas)
    CUTE_DSL_PTXAS_VERBOSE - Set to 1 for verbose output
"""

import os
import sys
import re
import ctypes
import subprocess
from pathlib import Path

import cutlass


CUTE_DSL_PTXAS_PATH = os.environ.get("CUTE_DSL_PTXAS_PATH", None)
VERBOSE = os.environ.get("CUTE_DSL_PTXAS_VERBOSE", "0") == "1"

_original_load_cuda_library = None
_user_wanted_ptx = False  # True if user originally set CUTE_DSL_KEEP_PTX=1


def _log(msg):
    if VERBOSE:
        print(f"[ptxas] {msg}", file=sys.stderr)


def _get_ptx(compiled_func) -> tuple[str, Path] | None:
    """Find and read PTX file, stripping null bytes."""
    func_name = getattr(compiled_func, "function_name", None)
    if not func_name:
        return None

    dump_dir = os.environ.get("CUTE_DSL_DUMP_DIR", Path.cwd())
    for ptx_path in Path(dump_dir).glob(f"*{func_name}*.ptx"):
        content = ptx_path.read_text().rstrip("\x00")
        if ".entry " in content and content.rstrip().endswith("}"):
            _log(f"Found PTX: {ptx_path}")
            return content, ptx_path
    return None


def _compile_ptx(ptx_path: Path, ptx_content: str) -> bytes:
    """Compile PTX to cubin using system ptxas."""
    # Extract arch from PTX
    match = re.search(r"\.target\s+(sm_\d+[a-z]?)", ptx_content)
    arch = match.group(1) if match else "sm_90a"

    # Write stripped content back if needed
    if ptx_path.read_text() != ptx_content:
        ptx_path.write_text(ptx_content)

    # Compile
    cubin_tmp = ptx_path.with_suffix(".cubin.tmp")
    try:
        assert CUTE_DSL_PTXAS_PATH is not None
        result = subprocess.run(
            [CUTE_DSL_PTXAS_PATH, f"-arch={arch}", "-O3", "-o", str(cubin_tmp), str(ptx_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ptxas failed: {result.stderr}")

        cubin_data = cubin_tmp.read_bytes()
        _log(f"Compiled {ptx_path.name} -> {len(cubin_data)} bytes ({arch})")

        # Save cubin if CUTE_DSL_KEEP_CUBIN is set
        if os.environ.get("CUTE_DSL_KEEP_CUBIN", "0") == "1":
            cubin_out = ptx_path.with_suffix(".cubin")
            cubin_out.write_bytes(cubin_data)
            _log(f"Saved: {cubin_out}")

        return cubin_data
    finally:
        cubin_tmp.unlink(missing_ok=True)


def _patched_load_cuda_library(self):
    """Replacement for _load_cuda_library that uses system ptxas."""

    result = _get_ptx(self)
    if not result:
        _log("PTX not found, falling back to embedded ptxas")
        return _original_load_cuda_library(self)

    ptx_content, ptx_path = result

    try:
        cubin = _compile_ptx(ptx_path, ptx_content)
    except Exception as e:
        _log(f"Compilation failed ({e}), falling back to embedded ptxas")
        return _original_load_cuda_library(self)

    # Load cubin
    import cuda.bindings.runtime as cuda_runtime

    err, library = cuda_runtime.cudaLibraryLoadData(cubin, None, None, 0, None, None, 0)
    if err != cuda_runtime.cudaError_t.cudaSuccess:
        _log(f"cudaLibraryLoadData failed ({err}), falling back to embedded ptxas")
        return _original_load_cuda_library(self)

    # Register kernels on all devices
    _, cuda_load_to_device = self._get_cuda_init_and_load()
    lib_ptr = ctypes.c_void_p(int(library))
    dev_id = ctypes.c_int32(0)
    err_val = ctypes.c_int32(0)
    args = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.pointer(lib_ptr), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(dev_id), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(err_val), ctypes.c_void_p),
    )

    for dev in range(self.num_devices):
        dev_id.value = dev
        cuda_load_to_device(args)
        if err_val.value != 0:
            _log("cuda_load_to_device failed, falling back to embedded ptxas")
            return _original_load_cuda_library(self)

    _log(f"Loaded kernel from {ptx_path.name}")

    # Delete PTX if user didn't originally want it kept
    if not _user_wanted_ptx:
        ptx_path.unlink(missing_ok=True)

    return [cuda_runtime.cudaLibrary_t(lib_ptr.value)]


def patch():
    """Install system ptxas hook. Call before importing cutlass."""
    global _original_load_cuda_library, _user_wanted_ptx

    assert CUTE_DSL_PTXAS_PATH is not None
    if not os.path.isfile(CUTE_DSL_PTXAS_PATH) or not os.access(CUTE_DSL_PTXAS_PATH, os.X_OK):
        raise RuntimeError(f"ptxas not found: {CUTE_DSL_PTXAS_PATH}")

    # Track if user originally wanted PTX kept
    _user_wanted_ptx = os.environ.get("CUTE_DSL_KEEP_PTX", "0") == "1"
    # os.environ['CUTE_DSL_KEEP_PTX'] = '1'
    assert os.environ.get("CUTE_DSL_KEEP_PTX", "0") == "1", (
        "Require CUTE_DSL_KEEP_PTX=1 to use system's ptxas"
    )

    cls = cutlass.cutlass_dsl.cuda_jit_executor.CudaDialectJitCompiledFunction
    _original_load_cuda_library = cls._load_cuda_library
    cls._load_cuda_library = _patched_load_cuda_library
    _log("Patch applied")
    return
