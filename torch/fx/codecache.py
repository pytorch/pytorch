"""Disk-based code cache for FX graph modules with stack trace support."""

import os
import re
import hashlib
import functools
from typing import Any, Optional
from types import ModuleType
from bisect import bisect_right

from torch._inductor.codecache import write_atomic, _reload_python_module


class FXCodeCache:
    """Cache for FX-generated code with hash-based naming."""

    modules: list[ModuleType] = []
    modules_no_attr: dict[str, ModuleType] = {}
    linemaps: dict[str, tuple[list[int], list[str]]] = {}

    @classmethod
    def write(cls, source_code: str) -> tuple[str, str]:
        """Write code to /tmp/fx_codegen with hash-based naming."""
        fx_dir = "/tmp/fx_codegen"
        os.makedirs(fx_dir, exist_ok=True)

        hash_val = hashlib.sha256(source_code.strip().encode("utf-8")).hexdigest()[:16]
        key = f"fx_{hash_val}"
        path = os.path.join(fx_dir, f"{key}.py")

        # Only write if file doesn't exist (cache reuse)
        if not os.path.exists(path):
            write_atomic(path, source_code, make_dirs=True)

        return key, path

    @classmethod
    def load_by_key_path(
        cls,
        key: str,
        path: str,
        linemap: Optional[list[tuple[int, str]]] = None,
        attrs: Optional[dict[str, Any]] = None,
    ) -> ModuleType:
        """Load module from disk and attach optional linemap and attributes."""
        if linemap is None:
            linemap = []

        if attrs is None and path in cls.modules_no_attr:
            return cls.modules_no_attr[path]

        mod = _reload_python_module(key, path, set_sys_modules=True)

        if linemap:
            lines, stack_traces = zip(*linemap)
            cls.linemaps[path] = (list(lines), list(stack_traces))

        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)

        if attrs is None:
            cls.modules_no_attr[path] = mod
        cls.modules.append(mod)

        return mod

    @classmethod
    @functools.cache
    def stack_frames_for_code(
        cls, path: str, lineno: int
    ) -> Optional[list[dict[str, Any]]]:
        """Map generated code line to original stack frames for error reporting."""
        if path not in cls.linemaps or not cls.linemaps[path]:
            return None

        lines, stack_traces = cls.linemaps[path]

        p = bisect_right(lines, lineno)
        if p == 0:
            return None

        stack_trace_str = stack_traces[p - 1]
        return _parse_stack_trace(stack_trace_str) if stack_trace_str else None

    @classmethod
    def cache_clear(cls, purge: bool = False) -> None:
        """Clear module cache. If purge=True, also delete files from disk."""
        if purge:
            for mod in cls.modules:
                try:
                    if mod.__file__:
                        os.remove(mod.__file__)
                except FileNotFoundError:
                    pass

        cls.modules.clear()
        cls.modules_no_attr.clear()
        cls.linemaps.clear()

    @classmethod
    def get_dir(cls) -> str:
        """Return /tmp/fx_codegen directory path."""
        return "/tmp/fx_codegen"


def _parse_stack_trace(stack_trace: str) -> list[dict[str, Any]]:
    """Parse FX stack trace string into list of frame dicts."""
    regex = r'File "(.+)", line (\d+), in (.+)\n'
    matches = re.findall(regex, stack_trace)
    return [{"filename": f, "line": int(l), "name": n} for f, l, n in reversed(matches)]
