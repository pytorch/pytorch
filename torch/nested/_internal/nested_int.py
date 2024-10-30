import torch
from typing import *
from torch.fx.experimental.sym_node import SymNode

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
    def __init__(self, t_id, cache, coeff):
        self.t_id = t_id
        self.cache = cache
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
        # Do we still need this?c
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
        return NestedIntNode(self.t_id, self.cache, self.coeff * other)

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

# import torch
# from typing import *
# from torch.fx.experimental.sym_node import SymNode


# # First PR:
# #
# lib = torch.library.Library("nested", "FRAGMENT")

# is_registered = False

# def _register_ops():
#     global is_registered

#     if is_registered:
#         return
#     is_registered = True

#     lib.define("assert_eq(Tensor lhs, Tensor rhs) -> ()")

#     def assert_eq_impl(lhs, rhs):
#         if not torch.eq(lhs, rhs).all().item():
#             raise RuntimeError("error")  # todo a better message

#     def assert_eq_meta(lhs, rhs):
#         # no-op
#         pass

#     lib.impl("assert_eq", assert_eq_impl, "CPU")
#     lib.impl("assert_eq", assert_eq_impl, "CUDA")
#     lib.impl("assert_eq", assert_eq_meta, "Meta")

#     from torch._higher_order_ops.effects import (
#         _EffectType,
#         _register_effectful_op,
#     )

#     # Make sure this does not get DCE'd
#     _register_effectful_op(
#         torch.ops.nested.assert_eq.default, _EffectType.ORDERED
#     )

# def assert_same_raggedness(j0, j1, msg):
#     j0.node.assert_same_raggedness(j1.node, msg)


# def _is_same_raggedness(lhs, rhs) -> Optional[bool]:
#     from torch._subclasses.fake_tensor import FakeTensor

#     assert isinstance(lhs, NestedIntNode) and isinstance(rhs, NestedIntNode)
#     if lhs.tensor is rhs.tensor:
#         return True
#     lhs_fake, rhs_fake = isinstance(lhs.tensor, FakeTensor), isinstance(rhs.tensor, FakeTensor)
#     if lhs_fake or rhs_fake:
#         from torch._dynamo.exc import unimplemented

#         unimplemented(
#             # Trigger a graph break if the tensors are not the same,
#             # Advise to the user that they should use the helper.
#             # In eager, shape comparison is fine. If they want their
#             # thing to compile, additional work is needed.
#             # Also advise that any custom operation should also use the
#             # helper.
#             "NestedInt comparison does not support ..."
#         )
#     else:
#         # Requires tensors to be on cpu
#         return torch.eq(lhs.tensor, rhs.tensor).all().item()

# # NestedInt eq can return True/False/None
# def _eq(lhs, rhs) -> bool:
#     if (
#         isinstance(lhs, NestedIntNode) and isinstance(rhs, NestedIntNode) and
#         lhs.coeff == rhs.coeff
#     ):
#         return _is_same_raggedness(lhs, rhs)
#     else:
#         return False

# def _ge(lhs, rhs, op_name) -> bool:
#     if isinstance(rhs, NestedIntNode) and isinstance(lhs, NestedIntNode):
#         # is it possible to return None?
#         is_same_raggedness = _is_same_raggedness(lhs, rhs)
#         if is_same_raggedness:
#             return lhs.coeff >= rhs.coeff
#         # Relation is indeterminate in eager too!
#         raise ValueError(f"{op_name}: relation is indeterminate")
#     elif isinstance(lhs, NestedIntNode):
#         if rhs.is_constant() and rhs.constant_int() <= 2:
#             return True
#         raise ValueError(f"{op_name}: relation is indeterminate")
#     elif isinstance(rhs, NestedIntNode):
#         if lhs.is_constant() and lhs.constant_int() < 2:
#             return False
#         raise ValueError(f"{op_name}: relation is indeterminate")
#     else:
#         raise ValueError(f"{op_name}: inputs unsupported")


# def mb_wrap_constant_bool(ret, *, negate=False):
#     if ret is not None:
#         if negate:
#             ret = not ret
#         return torch._C._get_constant_bool_symnode(ret)
#     return None



# class NestedIntNode:
#     def __init__(self, t_id, cache, *, coeff):
#         _register_ops()

#         self.t_id = t_id  # useful for keeping track of the version
#         # dummy tensor serving as the key
#         # but wait, this means... nested int is no longer disposable
#         # previously, offset is what matters
#         # nested_int_key is a tensor that... now needs to be stored
#         # on NT and passed around?
#         # but to the perspective of the user... why do I need to carry
#         # this extra thing around.
#         #

#         #
#         # if you have a cpu tensor, that's never the key
#         # if you have a lengths tensor, that's never the key
#         # your tensor, will always hold... offsets, and then
#         # 
#         #
#         #
#         self.tensor = torch.empty(0)
#         self.coeff = coeff

#     def get_tensor(self):
#         return self.tensor

#     def nested_int_coeff(self):
#         return self.coeff

#     def maybe_as_int(self):
#         return None

#     def is_int(self):
#         # ???
#         return True

#     def is_float(self):
#         return False

#     def is_bool(self):
#         return False

#     def is_nested_int(self):
#         # Do we still need this?c
#         return True

#     def clone(self):
#         return self

#     def str(self):
#         if self.coeff == 1:
#             return f"j{self.t_id}"
#         return f"{self.coeff}*j{self.t_id}"

#     def __str__(self):
#         return self.str()

#     def __repr__(self):
#         return self.str()

#     def mul(self, other) -> "NestedIntNode":
#         if other.is_constant():
#             other = other.constant_int()
#         else:
#             raise ValueError(f"unsupported: {type(other)}")
#         return NestedIntNode(self.t_id, self.tensor, coeff=self.coeff * other)

#     # TODO(soulitzer): Check SymInt returning None for cpp
#     def eq(self, other) -> Optional[SymNode]:
#         return mb_wrap_constant_bool(_eq(self, other))

#     def ne(self, other) -> Optional[SymNode]:
#         return mb_wrap_constant_bool(_eq(self, other), negate=True)

#     def gt(self, other) -> Optional[SymNode]:
#         return mb_wrap_constant_bool(_ge(other, self, "gt"), negate=True)

#     def lt(self, other) -> Optional[SymNode]:
#         return mb_wrap_constant_bool(_ge(self, other, "lt"), negate=True)

#     def le(self, other) -> Optional[SymNode]:
#         return mb_wrap_constant_bool(_ge(other, self, "le"))

#     def ge(self, other) -> Optional[SymNode]:
#         return mb_wrap_constant_bool(_ge(self, other, "ge"))

#     def neg(self) -> SymNode:
#         return self._neg()  # type: ignore[attr-defined]

#     def is_symbolic(self):
#         return False

#     def nested_int(self):
#         return self.t_id

#     def is_constant(self):
#         return False

#     def wrap_int(self, num):
#         assert type(num) is int
#         return torch._C._get_constant_int_symnode(num)
