import pathlib
import types
from collections.abc import Sequence

__all__ = ["build_and_import_extension", "compile_extension_module"]

def build_and_import_extension(
    modname: str,
    functions: Sequence[tuple[str, str, str]],
    *,
    prologue: str = "",
    build_dir: pathlib.Path | None = None,
    include_dirs: Sequence[str] | None = None,
    more_init: str = "",
) -> types.ModuleType: ...

#
def compile_extension_module(
    name: str,
    builddir: pathlib.Path,
    include_dirs: Sequence[str],
    source_string: str,
    libraries: Sequence[str] | None = None,
    library_dirs: Sequence[str] | None = None,
) -> pathlib.Path: ...
