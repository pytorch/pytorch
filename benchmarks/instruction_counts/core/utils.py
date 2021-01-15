import atexit
import re
import shutil
import tempfile
import textwrap
from typing import Iterator, List, Optional, Tuple, Union

from core.api import Setup, TimerArgs, GroupedTimerArgs
from core.types import Definition, FlatIntermediateDefinition, Label
from worker.main import CostEstimate


def _flatten(
    key_prefix: Label,
    sub_schema: Definition,
    result: FlatIntermediateDefinition
) -> None:
    for k, value in sub_schema.items():
        if isinstance(k, tuple):
            assert all(isinstance(ki, str) for ki in k)
            key_suffix: Label = k
        elif k is None:
            key_suffix = ()
        else:
            assert isinstance(k, str)
            key_suffix = (k,)

        key: Label = key_prefix + key_suffix
        if isinstance(value, (TimerArgs, GroupedTimerArgs)):
            assert key not in result, f"duplicate key: {key}"
            result[key] = value
        else:
            assert isinstance(value, dict)
            _flatten(key_prefix=key, sub_schema=value, result=result)


def flatten(schema: Definition) -> FlatIntermediateDefinition:
    result: FlatIntermediateDefinition = {}
    _flatten(key_prefix=(), sub_schema=schema, result=result)

    # Ensure that we produced a valid flat definition.
    for k, v in result.items():
        assert isinstance(k, tuple)
        assert all(isinstance(ki, str) for ki in k)
        assert isinstance(v, (TimerArgs, GroupedTimerArgs))
    return result


_TEMPDIR: Optional[str] = None
def get_temp_dir() -> str:
    global _TEMPDIR
    if _TEMPDIR is None:
        temp_dir = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, path=temp_dir)
        _TEMPDIR = temp_dir
    return _TEMPDIR
