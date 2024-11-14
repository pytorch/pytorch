# TODO(soulitzer): Decide on a plan here for nested int.
# The thing is to measure how much overhead are we really adding.

import torch
from typing import *
from torch.fx.experimental.sym_node import SymNode
from torch.nested._internal.metadata_cache import MetadataCache
from torch.nested._internal.utils import try_get_fake_mode


def _raggedness_same(a: "NestedIntNode", b: "NestedIntNode"):
    # Eventually we want:
    # 1. compare their ids
    # 2. then compare object identity of tensors in cache
    # 3. in eager actually compare the data
    #    - in compile, we just graph break, tell people to use the helper.
    # 4. if in eager, we can return true and do an device-side assert.

    # For now only do (1)
    return a.t_id == b.t_id


def _eq(lhs, rhs) -> bool:
    return (
        isinstance(lhs, NestedIntNode)
        and isinstance(rhs, NestedIntNode)
        and _raggedness_same(lhs, rhs)
        and lhs.coeff == rhs.coeff
    )


def _ge(lhs, rhs) -> bool:
    if isinstance(rhs, NestedIntNode) and isinstance(lhs, NestedIntNode):
        if _raggedness_same(lhs, rhs):
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
        raise ValueError(f"inputs unsupported")


class NestedIntNode:
    def __init__(self, cache: MetadataCache, coeff):
        self.t_id = cache.eq_id
        self.cache: MetadataCache = cache
        self.coeff = coeff

    def nested_int_cache(self):
        return self.cache

    def nested_int_coeff(self):
        return self.coeff

    def maybe_as_int(self):
        return None

    def is_int(self):
        # ???
        return True

    def is_float(self):
        return False

    def is_bool(self):
        return False

    def is_nested_int(self):
        # Do we still need this?
        return True

    def clone(self):
        return self

    def str(self):
        if self.coeff == 1:
            return f"j{self.t_id}"
        return f"{self.coeff}*j{self.t_id}"

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    def mul(self, other) -> "SymNode":
        if other.is_constant():
            other = other.constant_int()
        else:
            raise ValueError(f"unsupported: {type(other)}")
        return NestedIntNode(self.cache, self.coeff * other)

    def eq(self, other) -> "SymNode":
        return torch._C._get_constant_bool_symnode(_eq(self, other))

    def ne(self, other) -> "SymNode":
        return torch._C._get_constant_bool_symnode(not _eq(self, other))

    def gt(self, other) -> "SymNode":
        return torch._C._get_constant_bool_symnode(not _ge(other, self))

    def lt(self, other) -> "SymNode":
        return torch._C._get_constant_bool_symnode(not _ge(self, other))

    def le(self, other) -> "SymNode":
        return torch._C._get_constant_bool_symnode(_ge(other, self))

    def ge(self, other) -> "SymNode":
        return torch._C._get_constant_bool_symnode(_ge(self, other))

    def neg(self) -> "SymNode":
        return self._neg()  # type: ignore[attr-defined]

    def is_symbolic(self):
        return False

    def nested_int(self):
        return self.t_id

    def is_constant(self):
        return False

    def wrap_int(self, num):
        assert type(num) is int
        return torch._C._get_constant_int_symnode(num)

    def _graph_repr(self):
        return self.str()


def get_nested_symint(cache: MetadataCache, *, coeff=1):
    mb_fake_mode = try_get_fake_mode(cache)
    if mb_fake_mode is not None:
        # In compile, keep the same instance of nested int around
        return mb_fake_mode.get_nested_symint(cache) * coeff
    else:
        # In eager, always create a fresh nested int.
        return torch.SymInt(NestedIntNode(cache, coeff=coeff))
