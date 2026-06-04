"""scikit-build-core dynamic metadata provider for dependencies.

Runtime dependencies are declared as PEP 735 dependency-groups in pyproject.toml
under an "rt-" namespace: ``rt-base`` holds the deps every build needs, and one
``rt-<variant>`` group per accelerator (each ``include-group``-ing ``rt-base``)
adds that variant's extra runtime pins. Those same groups are declared mutually
conflicting under ``[tool.uv]``, so a single ``uv.lock`` carries one consistent
resolution fork per variant. This provider reads the selected group to populate
the wheel's dynamic ``[project.dependencies]`` (Requires-Dist):

  * ``PYTORCH_VARIANT=<sel>`` -> ``rt-<sel>`` (e.g. cu126, cu130, xpu; includes rt-base)
  * unset                    -> ``rt-base``  (the lock-time / sdist default)

``BUILD_PYTHON_ONLY`` still appends the libtorch wheel dependency. The legacy
``PYTORCH_EXTRA_INSTALL_REQUIREMENTS`` env var remains honored, additive on top
of ``rt-base``, for back-compat during the migration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

__all__ = ["dynamic_metadata"]

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _is_truthy(val: str | None) -> bool:
    return val is not None and val.upper() in ("ON", "1", "YES", "TRUE", "Y")


def _load_dependency_groups() -> dict[str, Any]:
    with open(_PYPROJECT, "rb") as fh:
        return tomllib.load(fh).get("dependency-groups", {})


def _resolve_group(name: str, groups: Mapping[str, Any]) -> list[str]:
    """Expand a PEP 735 dependency group to a flat list of requirement strings.

    Prefer packaging's resolver; fall back to a minimal include-group expander
    when it is unavailable (older packaging in an isolated build env)."""
    try:
        from packaging.dependency_groups import DependencyGroupResolver

        return [str(r) for r in DependencyGroupResolver(groups).resolve(name)]
    except Exception:
        # Any failure (missing/old packaging, API drift) -> self-contained fallback.
        return _resolve_group_fallback(name, groups, set())


def _resolve_group_fallback(
    name: str, groups: Mapping[str, Any], seen: set[str]
) -> list[str]:
    if name in seen:
        raise ValueError(f"cyclic dependency-group include detected at {name!r}")
    seen.add(name)
    out: list[str] = []
    for item in groups.get(name, []):
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and "include-group" in item:
            out.extend(_resolve_group_fallback(item["include-group"], groups, seen))
        else:
            raise ValueError(f"unsupported dependency-group item: {item!r}")
    return out


def _get_torch_version() -> str:
    import importlib.util

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

    groups = _load_dependency_groups()

    variant = os.environ.get("PYTORCH_VARIANT")
    group_name = f"rt-{variant}" if variant else "rt-base"
    if group_name not in groups:
        # Unknown selector: fall back to the common base runtime set.
        group_name = "rt-base"
    deps = _resolve_group(group_name, groups)

    # Back-compat: legacy pipe-separated override, additive on top of rt-base.
    if not variant:
        extra = os.environ.get("PYTORCH_EXTRA_INSTALL_REQUIREMENTS")
        if extra:
            deps.extend(r.strip() for r in extra.split("|") if r.strip())

    # BUILD_PYTHON_ONLY: add the libtorch wheel as a dependency.
    if _is_truthy(os.environ.get("BUILD_PYTHON_ONLY")):
        libtorch_pkg = os.environ.get("LIBTORCH_PACKAGE_NAME", "torch_no_python")
        deps.append(f"{libtorch_pkg}=={_get_torch_version()}")

    return deps
