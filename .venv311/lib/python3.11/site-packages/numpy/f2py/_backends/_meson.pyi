from collections.abc import Callable
from pathlib import Path
from typing import Final
from typing import Literal as L

from typing_extensions import override

from ._backend import Backend

class MesonTemplate:
    modulename: Final[str]
    build_template_path: Final[Path]
    sources: Final[list[str | Path]]
    deps: Final[list[str]]
    libraries: Final[list[str]]
    library_dirs: Final[list[str | Path]]
    include_dirs: Final[list[str | Path]]
    substitutions: Final[dict[str, str]]
    objects: Final[list[str | Path]]
    fortran_args: Final[list[str]]
    pipeline: Final[list[Callable[[], None]]]
    build_type: Final[str]
    python_exe: Final[str]
    indent: Final[str]

    def __init__(
        self,
        /,
        modulename: str,
        sources: list[Path],
        deps: list[str],
        libraries: list[str],
        library_dirs: list[str | Path],
        include_dirs: list[str | Path],
        object_files: list[str | Path],
        linker_args: list[str],
        fortran_args: list[str],
        build_type: str,
        python_exe: str,
    ) -> None: ...

    #
    def initialize_template(self) -> None: ...
    def sources_substitution(self) -> None: ...
    def deps_substitution(self) -> None: ...
    def libraries_substitution(self) -> None: ...
    def include_substitution(self) -> None: ...
    def fortran_args_substitution(self) -> None: ...

    #
    def meson_build_template(self) -> str: ...
    def generate_meson_build(self) -> str: ...

class MesonBackend(Backend):
    dependencies: list[str]
    meson_build_dir: L["bdir"]
    build_type: L["debug", "release"]

    def __init__(self, /, *args: object, **kwargs: object) -> None: ...
    def write_meson_build(self, /, build_dir: Path) -> None: ...
    def run_meson(self, /, build_dir: Path) -> None: ...
    @override
    def compile(self) -> None: ...
