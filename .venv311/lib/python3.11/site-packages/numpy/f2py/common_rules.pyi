from collections.abc import Mapping
from typing import Any, Final

from .__version__ import version

f2py_version: Final = version

def findcommonblocks(block: Mapping[str, object], top: int = 1) -> list[tuple[str, list[str], dict[str, Any]]]: ...
def buildhooks(m: Mapping[str, object]) -> tuple[dict[str, Any], str]: ...
