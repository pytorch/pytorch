import abc
from pathlib import Path
from typing import Any, Final

class Backend(abc.ABC):
    modulename: Final[str]
    sources: Final[list[str | Path]]
    extra_objects: Final[list[str]]
    build_dir: Final[str | Path]
    include_dirs: Final[list[str | Path]]
    library_dirs: Final[list[str | Path]]
    libraries: Final[list[str]]
    define_macros: Final[list[tuple[str, str | None]]]
    undef_macros: Final[list[str]]
    f2py_flags: Final[list[str]]
    sysinfo_flags: Final[list[str]]
    fc_flags: Final[list[str]]
    flib_flags: Final[list[str]]
    setup_flags: Final[list[str]]
    remove_build_dir: Final[bool]
    extra_dat: Final[dict[str, Any]]

    def __init__(
        self,
        /,
        modulename: str,
        sources: list[str | Path],
        extra_objects: list[str],
        build_dir: str | Path,
        include_dirs: list[str | Path],
        library_dirs: list[str | Path],
        libraries: list[str],
        define_macros: list[tuple[str, str | None]],
        undef_macros: list[str],
        f2py_flags: list[str],
        sysinfo_flags: list[str],
        fc_flags: list[str],
        flib_flags: list[str],
        setup_flags: list[str],
        remove_build_dir: bool,
        extra_dat: dict[str, Any],
    ) -> None: ...

    #
    @abc.abstractmethod
    def compile(self) -> None: ...
