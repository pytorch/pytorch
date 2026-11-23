"""Dynamic-metadata provider for the version field.

Delegates to tools/generate_torch_version.py which resolves the version from
(in order of precedence):
  1. PYTORCH_BUILD_VERSION / PYTORCH_BUILD_NUMBER env vars (release/nightly)
  2. PKG-INFO (sdist)
  3. version.txt + git SHA (local dev builds)
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

# _common is resolved at build time via scikit-build-core's provider path,
# not statically importable from the repo root.
from _common import get_torch_version  # pyrefly: ignore[missing-import]


if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["dynamic_metadata"]


def dynamic_metadata(
    settings: Mapping[str, Any],
    project: Mapping[str, Any],
) -> dict[str, Any]:
    if settings:
        msg = f"This provider takes no settings, got {sorted(settings)}"
        raise RuntimeError(msg)

    return {"version": get_torch_version()}
