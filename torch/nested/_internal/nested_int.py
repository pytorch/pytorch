import itertools
from typing import *  # noqa: F403

import torch
from torch.fx.experimental._constant_symnode import ConstantIntNode
from torch.nested._internal.tensor_registry import register_tensor, try_get_int
from torch.nested._internal.utils import flatten_nested_metadata_to_dict


__all__ = ["NestedIntNode"]


NestedIntMetaSpec = Tuple[frozenset[int], frozenset[int]]


def _any_id_equal(d1: NestedIntMetaSpec, d2: NestedIntMetaSpec) -> bool:
    # Note: [Nested tensor id equality]
    #
    # 1. Equality of j0 and j1 is determined by the object instances
    #    their caches (we only look at source fields, e.g.
    #    _{host,device}_{offsets,lengths}. The object instance is mapped
    #    to an int by TensorRegistry. Two distinct object instances can
    #    share the same int in the cases of .to().
    #
    # 2. j0 holds metadata in the format: Tuple[frozenset, frozenset]
    #    where the  first set represents the "tensor ids" of the offsets
    #    and and the second set represents the "tensor_ids" of the lengths.
    #    Comparison between j0 and j1 is by intersecting the corresponding
    #    sets. If either intersection is non-empty, then we return True.
    return bool((d1[0] & d2[0]) | (d1[1] & d2[1]))


def _eq(lhs: Any, rhs: Any) -> bool:
    return (
        isinstance(lhs, NestedIntNode)
        and isinstance(rhs, NestedIntNode)
        and _any_id_equal(lhs.metadata, rhs.metadata)
        and lhs.coeff == rhs.coeff
    )


def _ge(lhs: Any, rhs: Any) -> bool:
    if isinstance(rhs, NestedIntNode) and isinstance(lhs, NestedIntNode):
        if _any_id_equal(lhs.metadata, rhs.metadata):
            return lhs.coeff >= rhs.coeff
        raise ValueError("ge: relation is indeterminate")
    elif isinstance(lhs, NestedIntNode):
        if rhs.is_constant() and rhs.constant_int() <= 2:
            return True
        raise ValueError("ge: relation is indeterminate")
    elif isinstance(rhs, NestedIntNode):
        if lhs.is_constant() and lhs.constant_int() < 2:
            return False
        raise ValueError("ge: relation is indeterminate")
    else:
        raise ValueError("inputs unsupported")


def _get_tensor_ids(t: torch.Tensor) -> Tuple[frozenset, frozenset]:
    tmp: Tuple[set[int], set[int]] = (set(), set())

    for t_name, t_inner in flatten_nested_metadata_to_dict(
        t,
        only_source_fields=True,
        unwrap_functional_tensor=True,
    ).items():
        if (t_id := try_get_int(t_inner)) is None:
            t_id = register_tensor(t_inner)
        if "lengths" in t_name[-1]:
            tmp[0].add(t_id)
        elif "offsets" in t_name[-1]:
            tmp[1].add(t_id)
    ret = (frozenset(tmp[0]), frozenset(tmp[1]))
    return ret


class NestedIntNode:
    def __init__(self, cache: torch.Tensor, coeff: int):
        self.cache = cache
        self.metadata: NestedIntMetaSpec = _get_tensor_ids(cache)
        self.coeff = coeff
        # For the purpose of j-number printing, arbitrarily
        # choose the first tensor id in the dict.
        self.str_id = next(
            itertools.chain(iter(self.metadata[0]), iter(self.metadata[1]))
        )

    def nested_int_coeff(self) -> int:
        return self.coeff

    def nested_int_cache(self) -> Any:
        return self.cache

    def maybe_as_int(self) -> Optional[int]:
        return None

    def is_int(self) -> bool:
        return True

    def is_float(self) -> bool:
        return False

    def is_bool(self) -> bool:
        return False

    def is_nested_int(self) -> bool:
        return True

    def clone(self) -> "NestedIntNode":
        return self

    def _str(self) -> Any:
        if self.coeff == 1:
            return f"j{self.str_id}"
        return f"{self.coeff}*j{self.str_id}"

    def str(self) -> Any:
        return self._str()

    def __str__(self) -> Any:
        return self._str()

    def __repr__(self) -> Any:
        return self._str()

    def _graph_repr(self) -> Any:
        return self._str()

    def mul(self, other: Any) -> "NestedIntNode":
        if other.is_constant():
            other = other.constant_int()
        else:
            raise ValueError(f"unsupported: {type(other)}")
        return NestedIntNode(self.cache, self.coeff * other)

    def eq(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_eq(self, other))

    def ne(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _eq(self, other))

    def gt(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _ge(other, self))

    def lt(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _ge(self, other))

    def le(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_ge(other, self))

    def ge(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_ge(self, other))

    def is_symbolic(self) -> bool:
        return False

    def nested_int_meta(self) -> Tuple[frozenset, frozenset]:
        # Needs to be hashable
        return self.metadata

    def is_constant(self) -> bool:
        return False

    def wrap_int(self, num: int) -> ConstantIntNode:
        assert type(num) is int
        return ConstantIntNode(num)


def get_metadata(x: torch.SymInt) -> torch.Tensor:
    if isinstance(x.node, NestedIntNode):
        return x.node.nested_int_cache()
    else:
        return x.node.hint.node.nested_int_cache()
