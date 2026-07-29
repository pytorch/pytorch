import functools
import importlib.util
import logging
import platform
import subprocess
from pathlib import Path


log = logging.getLogger(__name__)


def _shared_library_unavailable_reason() -> str | None:
    flydsl_spec = importlib.util.find_spec("flydsl")
    if flydsl_spec is None or flydsl_spec.submodule_search_locations is None:
        return "missing optional dependency `flydsl`"

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
    # Shared with the eager gate, which runs during `import torch` and stays cheap.
    from torch._native.flydsl_utils import runtime_available as _runtime_installed

    if not _runtime_installed():
        return False

    # Inductor-only: codegen must know the runtime links before picking a template.
    reason = _shared_library_unavailable_reason()
    if reason is not None:
        log.debug("FlyDSL Inductor templates are unavailable: %s", reason)
        return False

    return True
