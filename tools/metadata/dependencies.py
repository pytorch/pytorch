"""scikit-build-core dynamic metadata provider for dependencies.

Replicates the old setup.py logic that reads PYTORCH_EXTRA_INSTALL_REQUIREMENTS
(pipe-separated PEP 508 dependency strings) and appends them to the base
dependency list.  Also handles BUILD_PYTHON_ONLY which adds a dependency on
the libtorch wheel package.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["dynamic_metadata"]

BASE_DEPENDENCIES = [
    "filelock",
    "typing-extensions>=4.10.0",
    "setuptools<82",
    "sympy>=1.13.3",
    "networkx>=2.5.1",
    "jinja2",
    "fsspec>=0.8.5",
]


def _is_truthy(val: str | None) -> bool:
    return val is not None and val.upper() in ("ON", "1", "YES", "TRUE", "Y")


def _get_torch_version() -> str:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "generate_torch_version",
        Path(__file__).resolve().parent.parent / "generate_torch_version.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load generate_torch_version.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_torch_version()


def dynamic_metadata(
    field: str,
    settings: Mapping[str, Any],
) -> list[str]:
    if field != "dependencies":
        msg = f"This provider only supports the 'dependencies' field, got {field!r}"
        raise RuntimeError(msg)

    deps = list(BASE_DEPENDENCIES)

    # BUILD_PYTHON_ONLY: add libtorch wheel as a dependency
    if _is_truthy(os.environ.get("BUILD_PYTHON_ONLY")):
        libtorch_pkg = os.environ.get("LIBTORCH_PACKAGE_NAME", "torch_no_python")
        version = _get_torch_version()
        deps.append(f"{libtorch_pkg}=={version}")

    # PYTORCH_EXTRA_INSTALL_REQUIREMENTS: pipe-separated PEP 508 strings
    extra = os.environ.get("PYTORCH_EXTRA_INSTALL_REQUIREMENTS")
    if extra:
        deps.extend(r.strip() for r in extra.split("|") if r.strip())

    return deps
