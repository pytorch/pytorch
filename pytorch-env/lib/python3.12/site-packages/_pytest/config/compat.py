from __future__ import annotations

import functools
from pathlib import Path
from typing import Any
from typing import Mapping
import warnings

import pluggy

from ..compat import LEGACY_PATH
from ..compat import legacy_path
from ..deprecated import HOOK_LEGACY_PATH_ARG


# hookname: (Path, LEGACY_PATH)
imply_paths_hooks: Mapping[str, tuple[str, str]] = {
    "pytest_ignore_collect": ("collection_path", "path"),
    "pytest_collect_file": ("file_path", "path"),
    "pytest_pycollect_makemodule": ("module_path", "path"),
    "pytest_report_header": ("start_path", "startdir"),
    "pytest_report_collectionfinish": ("start_path", "startdir"),
}


def _check_path(path: Path, fspath: LEGACY_PATH) -> None:
    if Path(fspath) != path:
        raise ValueError(
            f"Path({fspath!r}) != {path!r}\n"
            "if both path and fspath are given they need to be equal"
        )


class PathAwareHookProxy:
    """
    this helper wraps around hook callers
    until pluggy supports fixingcalls, this one will do

    it currently doesn't return full hook caller proxies for fixed hooks,
    this may have to be changed later depending on bugs
    """

    def __init__(self, hook_relay: pluggy.HookRelay) -> None:
        self._hook_relay = hook_relay

    def __dir__(self) -> list[str]:
        return dir(self._hook_relay)

    def __getattr__(self, key: str) -> pluggy.HookCaller:
        hook: pluggy.HookCaller = getattr(self._hook_relay, key)
        if key not in imply_paths_hooks:
            self.__dict__[key] = hook
            return hook
        else:
            path_var, fspath_var = imply_paths_hooks[key]

            @functools.wraps(hook)
            def fixed_hook(**kw: Any) -> Any:
                path_value: Path | None = kw.pop(path_var, None)
                fspath_value: LEGACY_PATH | None = kw.pop(fspath_var, None)
                if fspath_value is not None:
                    warnings.warn(
                        HOOK_LEGACY_PATH_ARG.format(
                            pylib_path_arg=fspath_var, pathlib_path_arg=path_var
                        ),
                        stacklevel=2,
                    )
                if path_value is not None:
                    if fspath_value is not None:
                        _check_path(path_value, fspath_value)
                    else:
                        fspath_value = legacy_path(path_value)
                else:
                    assert fspath_value is not None
                    path_value = Path(fspath_value)

                kw[path_var] = path_value
                kw[fspath_var] = fspath_value
                return hook(**kw)

            fixed_hook.name = hook.name  # type: ignore[attr-defined]
            fixed_hook.spec = hook.spec  # type: ignore[attr-defined]
            fixed_hook.__name__ = key
            self.__dict__[key] = fixed_hook
            return fixed_hook  # type: ignore[return-value]
