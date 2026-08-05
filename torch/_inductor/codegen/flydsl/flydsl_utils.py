import functools
import logging
from importlib.machinery import PathFinder
from importlib.util import find_spec
from pathlib import Path
from platform import system
from subprocess import run

from torch.backends import cuda as _cuda


log = logging.getLogger(__name__)
_pathfinder_find_spec = PathFinder.find_spec


def fits_int32_buffer_span(
    rows: int, row_stride: int | None, cols: int, itemsize: int
) -> bool:
    """Return whether a 2D tensor fits FlyDSL's 32-bit dimensions and offsets."""
    int32_max = (1 << 31) - 1
    return (
        0 < rows <= int32_max
        and 0 < cols <= int32_max
        and row_stride is not None
        and 0 <= row_stride <= int32_max
        and ((rows - 1) * row_stride + cols) * itemsize <= 1 << 32
    )


def _flydsl_runtime_unavailable_reason() -> str | None:
    flydsl_spec = find_spec("flydsl")
    if flydsl_spec is None or flydsl_spec.submodule_search_locations is None:
        return "missing optional dependency `flydsl`"

    # Query the package paths directly so this availability check does not
    # import flydsl as a side effect during regular torch imports.
    mlir_spec = _pathfinder_find_spec(
        "_mlir",
        list(flydsl_spec.submodule_search_locations),
    )
    if mlir_spec is None:
        return "missing optional dependency `flydsl._mlir`"

    if mlir_spec.submodule_search_locations:
        mlir_path = Path(next(iter(mlir_spec.submodule_search_locations)))
    elif mlir_spec.origin:
        mlir_path = Path(mlir_spec.origin).parent
    else:
        return "could not locate optional dependency `flydsl._mlir`"

    runtime_so = mlir_path / "_mlir_libs" / "libfly_jit_runtime.so"
    if not runtime_so.exists():
        return f"missing FlyDSL runtime shared library `{runtime_so}`"

    if system() == "Linux":
        try:
            ldd = run(
                ["ldd", str(runtime_so)],
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError as e:
            return f"could not inspect FlyDSL runtime shared library dependencies: {e}"

        ldd_output = f"{ldd.stdout}\n{ldd.stderr}"
        if ldd.returncode != 0 or "not found" in ldd_output:
            return (
                "unresolved FlyDSL runtime shared library dependencies: "
                + "; ".join(
                    line.strip()
                    for line in ldd_output.splitlines()
                    if "not found" in line
                )
            )

    return None


@functools.cache
def runtime_available() -> bool:
    import torch

    if not _cuda.is_built():
        return False

    if torch.version.hip is None:
        return False

    reason = _flydsl_runtime_unavailable_reason()
    if reason is not None:
        log.debug("FlyDSL Inductor templates are unavailable: %s", reason)
        return False

    return True
