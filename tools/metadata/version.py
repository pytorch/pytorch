"""scikit-build-core dynamic metadata provider for version.

Delegates to tools/generate_torch_version.py which resolves the version from
(in order of precedence):
  1. PYTORCH_BUILD_VERSION / PYTORCH_BUILD_NUMBER env vars (release/nightly)
  2. PKG-INFO (sdist)
  3. version.txt + git SHA (local dev builds)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["dynamic_metadata"]


def dynamic_metadata(
    field: str,
    settings: Mapping[str, Any],
) -> str:
    if field != "version":
        msg = f"This provider only supports the 'version' field, got {field!r}"
        raise RuntimeError(msg)

    spec = importlib.util.spec_from_file_location(
        "generate_torch_version",
        Path(__file__).resolve().parent.parent / "generate_torch_version.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load generate_torch_version.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_torch_version()
