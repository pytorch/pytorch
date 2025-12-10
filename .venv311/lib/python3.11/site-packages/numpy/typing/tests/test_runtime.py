"""Test the runtime usage of `numpy.typing`."""

from typing import (
    Any,
    NamedTuple,
    Union,  # pyright: ignore[reportDeprecated]
    get_args,
    get_origin,
    get_type_hints,
)

import pytest

import numpy as np
import numpy._typing as _npt
import numpy.typing as npt


class TypeTup(NamedTuple):
    typ: type
    args: tuple[type, ...]
    origin: type | None


NDArrayTup = TypeTup(npt.NDArray, npt.NDArray.__args__, np.ndarray)

TYPES = {
    "ArrayLike": TypeTup(npt.ArrayLike, npt.ArrayLike.__args__, Union),
    "DTypeLike": TypeTup(npt.DTypeLike, npt.DTypeLike.__args__, Union),
    "NBitBase": TypeTup(npt.NBitBase, (), None),
    "NDArray": NDArrayTup,
}


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_args(name: type, tup: TypeTup) -> None:
    """Test `typing.get_args`."""
    typ, ref = tup.typ, tup.args
    out = get_args(typ)
    assert out == ref


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_origin(name: type, tup: TypeTup) -> None:
    """Test `typing.get_origin`."""
    typ, ref = tup.typ, tup.origin
    out = get_origin(typ)
    assert out == ref


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_type_hints(name: type, tup: TypeTup) -> None:
    """Test `typing.get_type_hints`."""
    typ = tup.typ

    def func(a: typ) -> None: pass

    out = get_type_hints(func)
    ref = {"a": typ, "return": type(None)}
    assert out == ref


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_type_hints_str(name: type, tup: TypeTup) -> None:
    """Test `typing.get_type_hints` with string-representation of types."""
    typ_str, typ = f"npt.{name}", tup.typ

    def func(a: typ_str) -> None: pass

    out = get_type_hints(func)
    ref = {"a": typ, "return": type(None)}
    assert out == ref


def test_keys() -> None:
    """Test that ``TYPES.keys()`` and ``numpy.typing.__all__`` are synced."""
    keys = TYPES.keys()
    ref = set(npt.__all__)
    assert keys == ref


PROTOCOLS: dict[str, tuple[type[Any], object]] = {
    "_SupportsDType": (_npt._SupportsDType, np.int64(1)),
    "_SupportsArray": (_npt._SupportsArray, np.arange(10)),
    "_SupportsArrayFunc": (_npt._SupportsArrayFunc, np.arange(10)),
    "_NestedSequence": (_npt._NestedSequence, [1]),
}


@pytest.mark.parametrize("cls,obj", PROTOCOLS.values(), ids=PROTOCOLS.keys())
class TestRuntimeProtocol:
    def test_isinstance(self, cls: type[Any], obj: object) -> None:
        assert isinstance(obj, cls)
        assert not isinstance(None, cls)

    def test_issubclass(self, cls: type[Any], obj: object) -> None:
        if cls is _npt._SupportsDType:
            pytest.xfail(
                "Protocols with non-method members don't support issubclass()"
            )
        assert issubclass(type(obj), cls)
        assert not issubclass(type(None), cls)
