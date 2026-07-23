import functools
import importlib.machinery
import importlib.util
import logging
import platform
import subprocess
from pathlib import Path

from torch.backends import cuda as _cuda


log = logging.getLogger(__name__)


def _flydsl_runtime_unavailable_reason() -> str | None:
    flydsl_spec = importlib.util.find_spec("flydsl")
    if flydsl_spec is None or flydsl_spec.submodule_search_locations is None:
        return "missing optional dependency `flydsl`"

    # Query the package paths directly so this availability check does not
    # import flydsl as a side effect during regular torch imports.
    mlir_spec = importlib.machinery.PathFinder.find_spec(
        "_mlir",
        list(flydsl_spec.submodule_search_locations),
    )
    if mlir_spec is None:
        return "missing optional dependency `flydsl._mlir`"

    runtime_so = (
        Path(next(iter(flydsl_spec.submodule_search_locations)))
        / "_mlir"
        / "_mlir_libs"
        / "libfly_jit_runtime.so"
    )
    if not runtime_so.exists():
        return f"missing FlyDSL runtime shared library `{runtime_so}`"

    if platform.system() == "Linux":
        try:
            ldd = subprocess.run(
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

    if torch.version.hip is None:
        return False

    if not _cuda.is_built():
        return False

    reason = _flydsl_runtime_unavailable_reason()
    if reason is not None:
        log.debug("FlyDSL Inductor templates are unavailable: %s", reason)
        return False

    return True
