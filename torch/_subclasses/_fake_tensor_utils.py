from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Union

import torch
from torch import SymInt
from torch.fx.experimental.sym_node import SymNode
from torch.types import py_sym_types, PySymType


if TYPE_CHECKING:
    import sympy

    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from .fake_tensor import _DispatchCacheKey, _MetadataIntLike


@dataclass(frozen=True, slots=True)
class _DeconstructedSymNode:
    """
    Represents a SymNode without the associated ShapeEnv
    """

    # n.b. keep the same protocol as SymNode
    _expr: sympy.Expr
    pytype: type
    _hint: Optional[Union[int, float, bool]]
    constant: Optional[Union[int, float, bool]]
    fx_node: torch.fx.Node

    @staticmethod
    def from_node(node: SymNode) -> _DeconstructedSymNode:
        return _DeconstructedSymNode(
            node._expr,
            node.pytype,
            node._hint,
            node.constant,
            # pyrefly: ignore [bad-argument-type]
            node.fx_node,
        )

    def extract(self, shape_env: ShapeEnv) -> SymNode:
        return SymNode(
            self._expr, shape_env, self.pytype, self._hint, self.constant, self.fx_node
        )

    def __str__(self) -> str:
        return str(self._expr)

    def __repr__(self) -> str:
        return f"_DeconstructedSymNode{{{self._expr!r}, {self.pytype!r}, {self._hint!r}, {self.constant!r}, {self.fx_node!r}}}"

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    # _value_eq to match SymNode
    def _value_eq(self, other: object) -> bool:
        if isinstance(other, (SymNode, _DeconstructedSymNode)):
            return (
                self._expr == other._expr
                and self.pytype == other.pytype
                and self._hint == other._hint
                and self.constant == other.constant
                and self.fx_node == other.fx_node
            )
        else:
            return False

    # _value_hash to match SymNode
    def _value_hash(self) -> int:
        return hash((self._expr, self.pytype, self._hint, self.constant, self.fx_node))


@dataclass(frozen=True, slots=True)
class _DeconstructedSymType:
    """
    Represents a SymInt, SymFloat, SymBool without the associated ShapeEnv
    """

    ty: type[PySymType]
    node: _DeconstructedSymNode

    @staticmethod
    def from_sym_type(value: PySymType) -> _DeconstructedSymType:
        return _DeconstructedSymType(type(value), value.node)

    def extract(self, shape_env: ShapeEnv) -> PySymType:
        return self.ty(self.node.extract(shape_env))

    def __str__(self) -> str:
        return f"{self.ty}({self.node})"

    def __repr__(self) -> str:
        return f"_DeconstructedSymType({self.ty}, {self.node!r})"

    def __eq__(self, other: object) -> bool:
        return NotImplemented

    def __hash__(self) -> int:
        return NotImplemented


@dataclass(frozen=True, slots=True)
class _InputBackref:
    value: int


@dataclass(slots=True)
class _PySymInputStub:
    """
    Represents a SymInt in the cached key. Needed because SymInt doesn't
    support __eq__ or __hash__ directly.
    """

    # value can be:
    #   PySymType: This is the 'normal' SymInt value, wrapped so we can use
    #              hash/eq as value hash/eq (normally SymInt does object
    #              hash/eq).
    #   _DeconstructedSymType: This is used when storing the _PySymInputStub in
    #                          the cache to avoid cyclic ShapeEnv references.
    #   _InputBackref: This is a back-reference to a previous _PySymInputStub in
    #                  the key.
    value: Union[PySymType, _DeconstructedSymType, _InputBackref]

    def __init__(
        self, value: Union[PySymType, _DeconstructedSymType, _InputBackref]
    ) -> None:
        # For inputs (values in the `key`) we need to keep the PySymType intact
        # - this way if we need to reuse it as an output we can properly copy
        # the original value.
        self.value = value

    def strip_shape_env(self) -> None:
        if isinstance(self.value, py_sym_types):
            self.value = _DeconstructedSymType.from_sym_type(self.value)

    def extract(self, shape_env: ShapeEnv) -> PySymType:
        if isinstance(self.value, _DeconstructedSymType):
            return self.value.extract(shape_env)
        else:
            # We should never see an _InputBackref here - anyone extracting a
            # value should be pulling from the original entry (the one this
            # backref points at).
            assert not isinstance(self.value, _InputBackref)
            return self.value

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"_PySymInputStub({self.value!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PySymInputStub):
            return False
        elif isinstance(self.value, _InputBackref) or isinstance(
            other.value, _InputBackref
        ):
            return self.value == other.value
        else:
            return self.value.node._value_eq(other.value.node)

    def __hash__(self) -> int:
        if isinstance(self.value, _InputBackref):
            return hash(self.value)
        else:
            return self.value.node._value_hash()


@dataclass(slots=True)
class _SymIntOutputStub:
    """
    Represents a SymInt in the cached output.
    """

    # This is either an `int` which represents the index in the key to copy the
    # SymNode from or it's the deconstructed SymNode itself.
    value: Union[int, _DeconstructedSymNode]

    def __init__(self, value: SymInt, key_path: Optional[int]) -> None:
        if key_path is None:
            self.value = _DeconstructedSymNode.from_node(value.node)
        else:
            self.value = key_path

    def extract(self, key: _DispatchCacheKey, shape_env: ShapeEnv) -> SymInt:
        if isinstance(self.value, _DeconstructedSymNode):
            return SymInt(self.value.extract(shape_env))
        else:
            src = key.key[self.value]
            assert isinstance(src, _PySymInputStub) and isinstance(src.value, SymInt)
            return src.value

    def __repr__(self) -> str:
        return f"_SymIntOutputStub({self.value!r})"

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError


@dataclass(slots=True)
class _CacheKeyState:
    """
    State used while building our cache key.
    """

    # We track the SymNodes so when we get the output we can see if it exactly
    # matches one of the inputs so we can uncache it properly.
    sym_node_lookup: dict[int, int]  # id(SymNode) -> index

    # This is a list of all seen input sympy.Symbols. We use it when building
    # the cache entry to see if the output value has any symbols that we didn't
    # see on input. See _has_unrepresented_symbols().
    known_symbols: set[sympy.Symbol]

    # There are cases where we're asked to perform an op when we have no
    # ShapeEnv on the FakeTensorMode - but for SymNodes we MUST have a
    # ShapeEnv. So as we scan if we see a SymNode (with a ShapeEnv) we record it
    # here.
    shape_env: Optional[ShapeEnv]

    def __init__(self, shape_env: Optional[ShapeEnv] = None) -> None:
        self.sym_node_lookup = {}
        self.known_symbols = set()
        self.shape_env = shape_env

    def cache_on_shape_env(self) -> bool:
        """
        Returns true if the CacheKey needs to be cached on the ShapeEnv
        rather than the global cache.

        If our inputs contain a SymNode then we can't cache this operation on
        the global cache because the cached output will implicitly depend on
        guard values which might not be true on some other ShapeEnv. So unless
        we're also going to cache the guards we need to cache this operation on
        the ShapeEnv instead of globally.
        """
        return bool(self.sym_node_lookup)

    def convert_sym_int(self, result: list[object], arg: SymInt) -> None:
        node_id = id(arg.node)
        if node_id in self.sym_node_lookup:
            result.append(_InputBackref(self.sym_node_lookup[node_id]))
        else:
            self.sym_node_lookup[node_id] = len(result)
            self.known_symbols.update(arg.node.expr.free_symbols)
            if self.shape_env is None:
                self.shape_env = arg.node.shape_env
            result.append(_PySymInputStub(arg))

    def convert_output(self, arg: _MetadataIntLike) -> _MetadataIntLike:
        if isinstance(arg, SymInt):
            return _SymIntOutputStub(arg, self.sym_node_lookup.get(id(arg.node), None))
        else:
            return arg
