"""Shared helpers for the tools/metadata dynamic-metadata providers.

Sibling modules import this at top level: scikit-build-core resolves a
provider's imports through a meta-path finder that is only active while the
provider module itself is being imported, so a deferred (in-function) import
of this module would not resolve.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def get_torch_version() -> str:
    """Resolve the torch version via tools/generate_torch_version.py.

    Loaded by file path rather than imported, since tools/ is not on sys.path
    during the scikit-build-core metadata phase.
    """
    spec = importlib.util.spec_from_file_location(
        "generate_torch_version",
        Path(__file__).resolve().parent.parent / "generate_torch_version.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load generate_torch_version.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_torch_version()
