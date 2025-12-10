import argparse
import pprint
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from types import ModuleType
from typing import Any, Final, NotRequired, TypedDict, type_check_only

from typing_extensions import TypeVar, override

from .__version__ import version
from .auxfuncs import _Bool
from .auxfuncs import outmess as outmess

###

_KT = TypeVar("_KT", bound=Hashable)
_VT = TypeVar("_VT")

@type_check_only
class _F2PyDict(TypedDict):
    csrc: list[str]
    h: list[str]
    fsrc: NotRequired[list[str]]
    ltx: NotRequired[list[str]]

@type_check_only
class _PreparseResult(TypedDict):
    dependencies: list[str]
    backend: str
    modulename: str

###

MESON_ONLY_VER: Final[bool]
f2py_version: Final = version
numpy_version: Final = version
__usage__: Final[str]

show = pprint.pprint

class CombineIncludePaths(argparse.Action):
    @override
    def __call__(
        self,
        /,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ) -> None: ...

#
def run_main(comline_list: Iterable[str]) -> dict[str, _F2PyDict]: ...
def run_compile() -> None: ...
def main() -> None: ...

#
def scaninputline(inputline: Iterable[str]) -> tuple[list[str], dict[str, _Bool]]: ...
def callcrackfortran(files: list[str], options: dict[str, bool]) -> list[dict[str, Any]]: ...
def buildmodules(lst: Iterable[Mapping[str, object]]) -> dict[str, dict[str, Any]]: ...
def dict_append(d_out: MutableMapping[_KT, _VT], d_in: Mapping[_KT, _VT]) -> None: ...
def filter_files(
    prefix: str,
    suffix: str,
    files: Iterable[str],
    remove_prefix: _Bool | None = None,
) -> tuple[list[str], list[str]]: ...
def get_prefix(module: ModuleType) -> str: ...
def get_newer_options(iline: Iterable[str]) -> tuple[list[str], Any, list[str]]: ...

#
def f2py_parser() -> argparse.ArgumentParser: ...
def make_f2py_compile_parser() -> argparse.ArgumentParser: ...

#
def preparse_sysargv() -> _PreparseResult: ...
def validate_modulename(pyf_files: Sequence[str], modulename: str = "untitled") -> str: ...
