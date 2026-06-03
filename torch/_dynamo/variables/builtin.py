"""
Built-in function and type variable tracking for TorchDynamo's symbolic execution.

This module contains variable tracker classes for Python built-in functions, types,
and operations during graph compilation. It handles symbolic execution of:

- Built-in functions (len, getattr, isinstance, etc.)
- Type constructors (int, float, str, list, dict, etc.)
- Built-in operators and methods
- Special Python constructs (super, hasattr, etc.)

Key classes:
- BuiltinVariable: Tracks built-in functions and handles their execution
- TypeVariable: Manages type constructor calls and type checking
- SuperVariable: Handles super() calls in class hierarchies

These variable trackers ensure that built-in Python operations are correctly
handled during symbolic execution, either by executing them directly when safe
or by creating appropriate graph nodes when needed.
"""

import ast
import builtins
import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import sys
import types
import typing
import unittest
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NoReturn, TYPE_CHECKING

import torch
from torch._subclasses.meta_utils import is_sparse_any
from torch.overrides import BaseTorchFunctionMode
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import config, graph_break_hints, polyfills, variables
from ..exc import (
    ObservedAttributeError,
    ObservedUserStopIteration,
    raise_observed_exception,
    raise_type_error,
    unimplemented,
    Unsupported,
    UserError,
    UserErrorType,
)
from ..guards import GuardBuilder, install_guard
from ..replay_record import DummyModule
from ..source import (
    AttrSource,
    GetItemSource,
    GlobalSource,
    is_constant_source,
    Source,
    TypeSource,
)
from ..utils import (
    check_constant_args,
    check_numpy_ndarray_args,
    check_unspec_or_constant_args,
    check_unspec_python_args,
    dict_methods,
    extract_fake_example_value,
    get_fake_value,
    guard_if_dyn,
    is_tensor_getset_descriptor,
    is_wrapper_or_member_descriptor,
    istype,
    numpy_operator_wrapper,
    proxy_args_kwargs,
    raise_args_mismatch,
    specialize_symnode,
    str_methods,
    tensortype_to_dtype,
    unpack_iterable,
)
from .base import (
    AsPythonConstantNotImplementedError,
    AttrMutationKind,
    NO_SUCH_SUBOBJ,
    ValueMutationNew,
    VariableTracker,
)
from .constant import ConstantVariable, FakeIdVariable
from .dicts import (
    ConstDictVariable,
    DictItemsVariable,
    DictKeysVariable,
    DictViewVariable,
)
from .hashable import is_hashable
from .lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    SizeVariable,
    TupleIteratorVariable,
    TupleVariable,
)
from .misc import NullVariable, StringFormatVariable
from .object_protocol import (
    binary_iop,
    binary_op,
    generic_abs,
    generic_bool,
    generic_float,
    generic_getiter,
    generic_inplace_multiply,
    generic_int,
    generic_len,
    generic_multiply,
    generic_neg,
    generic_pos,
    generic_repr,
    maybe_get_python_type,
    pysequence_check,
    vt_add,
    vt_getitem,
    vt_identity_compare,
    vt_inplace_add,
)
from .sets import FrozensetVariable, SetVariable
from .tensor import (
    FakeItemVariable,
    supported_comparison_ops,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .user_defined import (
    is_data_descriptor,
    UserDefinedObjectVariable,
    UserDefinedVariable,
)


if TYPE_CHECKING:
    # Cyclic dependency...
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

log = logging.getLogger(__name__)


IN_PLACE_DESUGARING_MAP = {
    operator.iadd: operator.add,
    operator.isub: operator.sub,
    operator.imul: operator.mul,
    operator.ifloordiv: operator.floordiv,
    operator.itruediv: operator.truediv,
    operator.imod: operator.mod,
    operator.imatmul: operator.imatmul,
    operator.ilshift: operator.lshift,
    operator.irshift: operator.rshift,
    operator.ipow: operator.pow,
    operator.iand: operator.and_,
    operator.ior: operator.or_,
    operator.ixor: operator.xor,
}

_BUILTIN_CONSTANT_FOLDABLE_METHODS: dict[type, frozenset[str]] = {
    int: frozenset({"__new__", "from_bytes"}),
    bool: frozenset({"__new__", "from_bytes"}),
    float: frozenset({"fromhex", "hex"}),
}
if sys.version_info >= (3, 14):
    _BUILTIN_CONSTANT_FOLDABLE_METHODS[complex] = frozenset({"from_number"})


_HandlerCallback = Callable[
    ["InstructionTranslatorBase", typing.Any, typing.Any], VariableTracker | None
]
_TrackersType = type[VariableTracker] | tuple[type[VariableTracker], ...]
_OPERATOR_TO_DUNDER: dict[Callable[..., Any], str] = {
    operator.eq: "__eq__",
    operator.ne: "__ne__",
    operator.lt: "__lt__",
    operator.le: "__le__",
    operator.gt: "__gt__",
    operator.ge: "__ge__",
}


bin_ops = (
    operator.pow,
    operator.mul,
    operator.matmul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.add,
    operator.lt,
    operator.gt,
    operator.ge,
    operator.le,
    operator.ne,
    operator.eq,
    operator.sub,
    operator.ipow,
    operator.imul,
    operator.imatmul,
    operator.ifloordiv,
    operator.itruediv,
    operator.imod,
    operator.iadd,
    operator.isub,
)

bin_int_ops = (
    operator.and_,
    operator.or_,
    operator.xor,
    operator.iand,
    operator.ixor,
    operator.ior,
)

un_int_ops = (operator.invert,)

tensor_and_int_ops = (
    operator.lshift,
    operator.rshift,
    operator.ilshift,
    operator.irshift,
    operator.getitem,
)

un_ops = (
    operator.abs,
    operator.pos,
    operator.neg,
    operator.not_,  # Note: this has a local scalar dense call
    operator.length_hint,
)

_SET_LIKE_OP_SUPPORT: tuple[type[VariableTracker], ...] = (
    DictItemsVariable,
    DictKeysVariable,
    SetVariable,
    UserDefinedObjectVariable,
)

BUILTIN_TO_TENSOR_FN_MAP: dict[Callable[..., Any], Callable[..., Any]] = {}

# These functions represent the r* versions of the above ops
# Basically, if __add__(1, Tensor) is called, it is translated
# to __radd__(Tensor, 1).
# In the builtin var, we check if there is a tensor in the first args position,
# if not, we swap the args and use the r* version of the op.
BUILTIN_TO_TENSOR_RFN_MAP: dict[Callable[..., Any], Callable[..., Any]] = {}

# Sentinel for `inspect.getattr_static` lookups that must distinguish
# "attribute absent" from "attribute is None" (e.g. `__reversed__ = None`
# opt-out).
_MISSING_SENTINEL = object()


def populate_builtin_to_tensor_fn_map() -> None:
    global BUILTIN_TO_TENSOR_FN_MAP
    if len(BUILTIN_TO_TENSOR_FN_MAP) > 0:
        # Only populate once; after there are elements present no need to
        # repopulate
        return
    most_recent_func: Callable[..., Any] | None = None

    class GetMethodMode(BaseTorchFunctionMode):
        """
        Mode to extract the correct methods from torch function invocations
        (Used to get the correct torch.Tensor methods from builtins)
        """

        def __torch_function__(
            self,
            func: Callable[..., Any],
            types: Any,
            args: Sequence[Any] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            kwargs = kwargs or {}
            nonlocal most_recent_func
            most_recent_func = func
            return func(*args, **kwargs)

    inp0 = torch.ones(1)
    inp1 = torch.ones(1)
    inp0_int = torch.ones(1, dtype=torch.int32)
    inp1_int = torch.ones(1, dtype=torch.int32)
    with GetMethodMode():
        setups_and_oplists: list[tuple[Callable[..., Any], Iterable[Any]]] = [
            (lambda o: o(inp0), un_ops),
            (lambda o: o(inp0_int), un_int_ops),
            (lambda o: o(inp0, inp1), bin_ops),
            (lambda o: o(inp0_int, inp1_int), bin_int_ops),
            (lambda o: o(inp0_int, 0), tensor_and_int_ops),
        ]
        for setup_fn, op_list in setups_and_oplists:
            for op in op_list:
                setup_fn(op)
                if most_recent_func is None:
                    raise AssertionError(
                        f"most_recent_func is None after setup for op {op}"
                    )
                BUILTIN_TO_TENSOR_FN_MAP[op] = most_recent_func

        # gather the reverse functions
        rsetups_and_oplists: list[tuple[Callable[..., Any], Iterable[Any]]] = [
            (
                lambda o: o(1, inp1),
                bin_ops,
            ),  # Get r* ops, (ex. __sub__(int, Tensor) -> __rsub__(Tensor, int))
            (lambda o: o(1, inp1_int), bin_int_ops),
            (lambda o: o(0, inp0_int), tensor_and_int_ops),
        ]

        rskips = {operator.matmul, operator.imatmul, operator.getitem}
        for setup_fn, op_list in rsetups_and_oplists:
            for op in op_list:
                if op in rskips:
                    continue
                setup_fn(op)
                if most_recent_func is None:
                    raise AssertionError(
                        f"most_recent_func is None after setup for reverse op {op}"
                    )
                if most_recent_func != BUILTIN_TO_TENSOR_FN_MAP[op]:
                    BUILTIN_TO_TENSOR_RFN_MAP[op] = most_recent_func


class BaseBuiltinVariable(VariableTracker):
    """
    Common base class for all builtin variable trackers (BuiltinVariable,
    DictBuiltinVariable, IterBuiltinVariable, and future specialized builtins).

    Provides shared implementations for guard installation, hasattr tracing,
    and Python-level hashability/equality.

    Specialized subclasses (e.g. DictBuiltinVariable) set `_fn` as a class
    attribute. BuiltinVariable stores the callable on the instance as `self.fn`
    and overrides as_python_constant / reconstruct / var_getattr accordingly.
    """

    _fn: Any = None

    @classmethod
    def create_with_source(cls, value: Any, source: Source) -> "BaseBuiltinVariable":
        install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        return cls(source=source)

    def as_python_constant(self) -> Any:
        return self._fn

    def reconstruct(self, codegen: "PyCodegen") -> None:
        name = self.as_python_constant().__name__
        if name in codegen.tx.f_globals:
            raise AssertionError("shadowed global")
        codegen.append_output(codegen.create_load_global(name, add=True))

    def var_getattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        source = self.source and AttrSource(self.source, name)
        attr = getattr(self._fn, name, None)
        return variables.GetAttrVariable(
            self, name, py_type=type(attr) if attr is not None else None, source=source
        )

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        return VariableTracker.build(tx, hasattr(self.as_python_constant(), name))  # type: ignore[return-value]

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # CPython meth_hash: https://github.com/python/cpython/blob/e76aa128fe/Objects/methodobject.c#L319
        return hash(self.as_python_constant()), False

    def richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def is_python_equal(self, other: object) -> bool:
        return isinstance(other, BaseBuiltinVariable) and (
            self.as_python_constant() is other.as_python_constant()  # type: ignore[union-attr]
        )

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__repr__" and len(args) == 1 and not kwargs:
            arg = args[0]
            if self.as_python_constant() is object and isinstance(
                arg, variables.UserDefinedObjectVariable
            ):
                return VariableTracker.build(tx, object.__repr__(arg.value))
            if self.as_python_constant() is type:
                if isinstance(arg, variables.UserDefinedClassVariable):
                    return VariableTracker.build(tx, type.__repr__(arg.value))
                if arg.is_python_constant() and isinstance(
                    arg.as_python_constant(), type
                ):
                    return VariableTracker.build(
                        tx, type.__repr__(arg.as_python_constant())
                    )
            return generic_repr(tx, arg)
        return super().call_method(tx, name, args, kwargs)


class BuiltinVariable(BaseBuiltinVariable):
    """
    A VariableTracker that represents a built-in value (functions and operators).
    A lot of the code here assumes it will be a function object.

    The BuiltinVariable class wraps Python built-in functions (like len, isinstance, etc.)
    and operators (like +, -, *, etc.) to enable symbolic execution during tracing. This allows
    Dynamo to properly handle these operations when converting Python code to FX graphs while
    maintaining correct semantics and enabling optimizations.
    """

    _SENTINEL = object()
    _nonvar_fields = {
        "fn",
        *VariableTracker._nonvar_fields,
    }

    @classmethod
    def create_with_source(cls, value: Any, source: Source) -> "BuiltinVariable":
        install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        return cls(value, source=source)

    @staticmethod
    @functools.cache
    def _constant_fold_functions() -> set[Callable[..., Any]]:
        fns: set[Callable[..., Any]] = {
            abs,
            all,
            any,
            ascii,
            bin,
            bool,
            callable,
            chr,
            complex,
            divmod,
            float,
            format,
            getattr,
            hex,
            int,
            len,
            max,
            min,
            oct,
            ord,
            pow,
            repr,
            round,
            str,
            str.format,
            sum,
            type,
            operator.abs,
            operator.pos,
            operator.neg,
            operator.not_,
            operator.truth,
            operator.invert,
            operator.pow,
            operator.mul,
            operator.matmul,
            operator.floordiv,
            operator.truediv,
            operator.mod,
            operator.add,
            operator.sub,
            operator.getitem,
            operator.length_hint,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            operator.ipow,
            operator.imul,
            operator.imatmul,
            operator.ifloordiv,
            operator.itruediv,
            operator.imod,
            operator.iadd,
            operator.isub,
            operator.ilshift,
            operator.irshift,
            operator.iand,
            operator.ixor,
            operator.ior,
            operator.index,
        }
        from .tensor import supported_comparison_ops

        fns.update(supported_comparison_ops.values())
        fns.update(x for x in math.__dict__.values() if isinstance(x, type(math.sqrt)))
        return fns

    def can_constant_fold_through(self) -> bool:
        return self.fn in self._constant_fold_functions()

    @staticmethod
    @functools.cache
    def _fx_graph_functions() -> set[Callable[..., Any]]:
        fns = {
            operator.abs,
            operator.pos,
            operator.neg,
            operator.not_,
            operator.invert,
            operator.pow,
            operator.mul,
            operator.matmul,
            operator.floordiv,
            operator.truediv,
            operator.mod,
            operator.add,
            operator.lt,
            operator.gt,
            operator.ge,
            operator.le,
            operator.ne,
            operator.eq,
            operator.sub,
            operator.length_hint,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            operator.ipow,
            operator.imul,
            operator.imatmul,
            operator.ifloordiv,
            operator.itruediv,
            operator.getitem,
            operator.imod,
            operator.iadd,
            operator.isub,
            operator.ilshift,
            operator.irshift,
            operator.iand,
            operator.ixor,
            operator.ior,
        }
        return fns  # type: ignore[return-value]

    @staticmethod
    @functools.cache
    def _binops() -> dict[
        Callable[..., object], tuple[list[str], Callable[..., object]]
    ]:
        # function -> ([forward name, reverse name, in-place name], in-place op)
        fns: dict[Callable[..., object], tuple[list[str], Callable[..., object]]] = {
            operator.truediv: (
                ["__truediv__", "__rtruediv__", "__itruediv__"],
                operator.itruediv,
            ),
            operator.floordiv: (
                ["__floordiv__", "__rfloordiv__", "__ifloordiv__"],
                operator.ifloordiv,
            ),
            operator.mod: (["__mod__", "__rmod__", "__imod__"], operator.imod),
            pow: (["__pow__", "__rpow__", "__ipow__"], operator.ipow),
            operator.pow: (["__pow__", "__rpow__", "__ipow__"], operator.ipow),
            operator.xor: (["__xor__", "__rxor__", "__ixor__"], operator.xor),
            # NB: The follow binary operators are not supported for now, since the
            # corresponding magic methods aren't defined on SymInt / SymFloat:
            # operator.matmul
            # divmod
            # operator.and_
            # operator.or_
        }
        return fns

    @staticmethod
    @functools.cache
    def _binop_handlers() -> dict[
        Callable[..., object],
        list[
            tuple[
                tuple[
                    type[VariableTracker],
                    _TrackersType,
                ],
                _HandlerCallback,
            ]
        ],
    ]:
        # Multiple dispatch mechanism defining custom binop behavior for certain type
        # combinations. Handlers are attempted in order, and will be used if the type checks
        # match. They are expected to have the signature:
        # fn(tx, arg0: VariableTracker, arg1: VariableTracker) -> VariableTracker
        from .functions import BaseUserFunctionVariable
        from .nn_module import NNModuleVariable
        from .tensor import supported_const_comparison_ops
        from .torch import BaseTorchVariable
        from .user_defined import UserDefinedVariable

        # Override table contains: op_fn -> [list of handlers]
        op_handlers: dict[Any, list[Any]] = {}
        for (
            op,
            (magic_method_names, in_place_op),
        ) in BuiltinVariable._binops().items():
            op_handlers[op] = []
            op_handlers[in_place_op] = []

            forward_name, reverse_name, inplace_name = magic_method_names

            # User-defined args (highest precedence)
            def user_defined_handler(
                tx: "InstructionTranslatorBase",
                a: VariableTracker,
                b: VariableTracker,
                *,
                forward_name: str = forward_name,
                reverse_name: str = reverse_name,
            ) -> VariableTracker:
                # Manually handle reversing logic if needed (e.g. call __radd__)

                # TODO: If we expand this to handle tensor args, we need to manually
                # handle cases like this:
                #
                # class A(int):
                #     def __radd__(self, other):
                #         print("woof")
                # torch.randn(3) + A(3)
                #
                # In this example, A.__radd__() is not called -> nothing is printed, because
                # Tensor.__add__ only does a subtype test against int, ignoring the subclass.
                # To be fully correct, we should not call A.__radd__() here, and there may be
                # other cases to reason about and add exceptions for.
                if isinstance(a, UserDefinedVariable):
                    return a.call_method(tx, forward_name, [b], {})
                else:
                    return b.call_method(tx, reverse_name, [a], {})

            op_handlers[op].append(
                ((UserDefinedVariable, VariableTracker), user_defined_handler)
            )
            op_handlers[op].append(
                ((VariableTracker, UserDefinedVariable), user_defined_handler)
            )

            def user_defined_inplace_handler(
                tx: "InstructionTranslatorBase",
                a: VariableTracker,
                b: VariableTracker,
                *,
                forward_name: str = inplace_name,
            ) -> VariableTracker:
                return a.call_method(tx, forward_name, [b], {})

            op_handlers[in_place_op].append(
                ((UserDefinedVariable, VariableTracker), user_defined_inplace_handler)
            )
            op_handlers[in_place_op].append(
                ((VariableTracker, UserDefinedVariable), user_defined_inplace_handler)
            )

            # Dynamic shape args
            def dynamic_handler(
                tx: "InstructionTranslatorBase",
                a: VariableTracker,
                b: VariableTracker,
                *,
                fn: Callable[..., Any] = op,
            ) -> VariableTracker:
                from .builder import wrap_fx_proxy

                return wrap_fx_proxy(
                    tx,
                    tx.output.create_proxy(
                        "call_function", fn, *proxy_args_kwargs([a, b], {})
                    ),
                )

            op_handlers[op].append(
                ((SymNodeVariable, VariableTracker), dynamic_handler)
            )
            op_handlers[op].append(
                ((VariableTracker, SymNodeVariable), dynamic_handler)
            )

            # NB: Prefer out-of-place op when calling in-place op to generate valid graph
            op_handlers[in_place_op].append(
                ((SymNodeVariable, VariableTracker), dynamic_handler)
            )
            op_handlers[in_place_op].append(
                ((VariableTracker, SymNodeVariable), dynamic_handler)
            )

        # Special cases - lower precedence but still prefer these over constant folding

        # List-like addition (e.g. [1, 2] + [3, 4])
        def tuple_add_handler(
            tx: "InstructionTranslatorBase", a: BaseListVariable, b: VariableTracker
        ) -> VariableTracker:
            return TupleVariable([*a.items, *b.unpack_var_sequence(tx)])

        def size_add_handler(
            tx: "InstructionTranslatorBase", a: BaseListVariable, b: VariableTracker
        ) -> VariableTracker:
            return SizeVariable([*a.items, *b.unpack_var_sequence(tx)])

        def create_cmp_op_handlers(
            op: Callable[..., Any],
        ) -> list[tuple[tuple[_TrackersType, _TrackersType], _HandlerCallback]]:
            def compare_by_value(
                tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
            ) -> VariableTracker:
                try:
                    return VariableTracker.build(tx, op(a.value, b.value))  # type: ignore[attr-defined]
                except TypeError as exc:
                    raise_observed_exception(
                        type(exc),
                        tx,
                        args=list(exc.args),
                    )

            result: list[
                tuple[
                    tuple[
                        _TrackersType,
                        _TrackersType,
                    ],
                    _HandlerCallback,
                ]
            ] = [((ConstantVariable, ConstantVariable), compare_by_value)]

            if op in _OPERATOR_TO_DUNDER:
                # For constants, speedup the comparison instead of going
                # through generic_richcompare. Removing this line causes
                # major regression for pr time benchmark - add_loop_eager.
                result = [
                    ((ConstantVariable, ConstantVariable), compare_by_value),
                ]

                op_var = BuiltinVariable(op)
                # Special handling of SymNode variable
                result.extend(
                    [
                        (
                            (SymNodeVariable, VariableTracker),
                            op_var._comparison_with_symnode,
                        ),
                        (
                            (VariableTracker, SymNodeVariable),
                            op_var._comparison_with_symnode,
                        ),
                    ]
                )

                # COMPARE_OP (a == b) dispatches through generic_richcompare,
                # which implements do_richcompare via richcompare_impl slots.
                # See object_protocol.py for details.
                dunder = _OPERATOR_TO_DUNDER[op]

                def handler(
                    tx: "InstructionTranslatorBase",
                    a: VariableTracker,
                    b: VariableTracker,
                ) -> VariableTracker:
                    from .object_protocol import generic_richcompare

                    return generic_richcompare(tx, a, b, dunder)

                result.append(((VariableTracker, VariableTracker), handler))
                return result

            result = [((ConstantVariable, ConstantVariable), compare_by_value)]

            if op in supported_const_comparison_ops.values() and op.__name__.startswith(
                "is_"
            ):
                # Tensor is None, List is not None, etc
                none_result = op(object(), None)

                def never(
                    tx: "InstructionTranslatorBase",
                    a: VariableTracker,
                    b: VariableTracker,
                ) -> VariableTracker:
                    return VariableTracker.build(tx, none_result)

                obj_op_none = never
                none_op_obj = never

                types_that_are_never_none = (
                    TensorVariable,
                    SymNodeVariable,
                    NNModuleVariable,
                    BaseListVariable,
                    UserDefinedVariable,
                    BaseUserFunctionVariable,
                    ConstDictVariable,
                    BaseTorchVariable,
                )
                result.extend(
                    [
                        (
                            (types_that_are_never_none, ConstantVariable),
                            obj_op_none,
                        ),
                        (
                            (ConstantVariable, types_that_are_never_none),
                            none_op_obj,
                        ),
                    ]
                )

                op_var = BuiltinVariable(op)
                result.extend(
                    [
                        (
                            (TensorVariable, VariableTracker),
                            op_var._comparison_with_tensor,
                        ),
                        (
                            (VariableTracker, TensorVariable),
                            op_var._comparison_with_tensor,
                        ),
                        (
                            (SymNodeVariable, VariableTracker),
                            op_var._comparison_with_symnode,
                        ),
                        (
                            (VariableTracker, SymNodeVariable),
                            op_var._comparison_with_symnode,
                        ),
                    ]
                )

                def handle_is(
                    tx: "InstructionTranslatorBase",
                    left: VariableTracker,
                    right: VariableTracker,
                ) -> VariableTracker | None:
                    result = vt_identity_compare(left, right)
                    if result is None:
                        return None
                    is_same = result.as_python_constant()
                    return VariableTracker.build(
                        tx, is_same if op.__name__ == "is_" else not is_same
                    )

                result.append(((VariableTracker, VariableTracker), handle_is))  # type: ignore[arg-type]

            return result

        for op in supported_comparison_ops.values():
            if not callable(op):
                raise AssertionError(f"comparison op {op} is not callable")
            if op in op_handlers:
                raise AssertionError(f"duplicate handler for op {op}")
            op_handlers[op] = create_cmp_op_handlers(op)

        return op_handlers

    @staticmethod
    def _find_binop_handler(
        op: Callable[..., Any], a_type: type[VariableTracker], b_type: type
    ) -> list[_HandlerCallback] | None:
        handlers = BuiltinVariable._binop_handlers().get(op)
        if handlers is None:
            return None

        matches = []
        for (type1, type2), handler in handlers:
            if issubclass(a_type, type1) and issubclass(b_type, type2):
                matches.append(handler)
        return matches

    def can_insert_in_graph(self) -> bool:
        return self.fn in self._fx_graph_functions()

    # Builtins that have been promoted to their own VT classes. Creating a
    # BuiltinVariable for these is a bug; use the specialized class instead.
    MUST_USE_SPECIALIZED: frozenset[Any] = frozenset(
        {dict, getattr, hasattr, iter, list, setattr}
    )

    def __init__(self, fn: Any, **kwargs: Any) -> None:
        if fn in self.MUST_USE_SPECIALIZED:
            raise AssertionError(
                f"Use the specialized VT class for {fn!r}, not BuiltinVariable. "
                f"E.g. DictBuiltinVariable for dict."
            )
        super().__init__(**kwargs)
        self.fn = fn

    def __repr__(self) -> str:
        if self.fn is None:
            name = "None"
        else:
            name = self.fn.__name__

        return f"{self.__class__.__name__}({name})"

    def as_python_constant(self) -> Any:
        return self.fn

    def get_real_python_backed_value(self) -> Any:
        return self.fn

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # BuiltinVariable wraps built-in types like list, tuple, dict.
        # type(self.fn).__or__(self.fn, other_val) delegates to CPython's
        # _Py_union_type_or for type unions (e.g. list | tuple).
        # https://github.com/python/cpython/blob/v3.13.0/Objects/typeobject.c#L6028-L6030 (type_as_number.nb_or = _Py_union_type_or)
        # https://github.com/python/cpython/blob/3.13/Objects/unionobject.c#L162 (_Py_union_type_or)
        if not isinstance(self.fn, type):
            return VariableTracker.build(tx, NotImplemented)
        try:
            other_val = other.as_python_constant()
        except NotImplementedError:
            return VariableTracker.build(tx, NotImplemented)
        # pyrefly: ignore[bad-argument-count]
        result = type(self.fn).__or__(self.fn, other_val)
        if result is NotImplemented:
            return VariableTracker.build(tx, NotImplemented)
        return VariableTracker.build(tx, result)

    def as_proxy(self) -> Any:
        DTYPE = {
            bool: torch.bool,
            int: torch.int64,
            float: torch.float64,
        }
        if self.fn in DTYPE:
            return DTYPE[self.fn]
        return super().as_proxy()

    def reconstruct(self, codegen: "PyCodegen") -> None:
        name = self.fn.__name__
        if self.fn.__module__ != "builtins":
            raise AssertionError(f"Expected builtins module, got {self.fn.__module__}")
        if name in codegen.tx.f_globals:
            raise AssertionError("shadowed global")
        codegen.append_output(codegen.create_load_global(name, add=True))

    def constant_args(self, *args: VariableTracker, **kwargs: VariableTracker) -> bool:
        return check_constant_args(args, kwargs)

    def tensor_args(self, *args: VariableTracker) -> bool:
        any_tensor = False
        for arg in args:
            if isinstance(arg, variables.GetAttrVariable):
                return False
            any_tensor = any_tensor or arg.is_tensor()
        return any_tensor

    def tensor_args_type(self, arg_types: list[type]) -> bool:
        any_tensor = False
        for arg_type in arg_types:
            if issubclass(arg_type, variables.GetAttrVariable):
                return False
            any_tensor = any_tensor or issubclass(arg_type, variables.TensorVariable)
        return any_tensor

    def python_and_tensor_constant_only(
        self, *args: VariableTracker, **kwargs: VariableTracker
    ) -> bool:
        tensor_args = []
        non_tensor_args = []
        for i in itertools.chain(args, kwargs.values()):
            if i.is_tensor():
                tensor_args.append(i)
            else:
                non_tensor_args.append(i)
        return all(
            is_constant_source(t.source) if t.source is not None else False
            for t in tensor_args
        ) and self.constant_args(*non_tensor_args)

    @staticmethod
    def unwrap_unspec_args_kwargs(
        args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> tuple[list[Any], dict[str, Any]]:
        return [x.as_python_constant() for x in args], {
            k: v.as_python_constant() for k, v in kwargs.items()
        }

    def has_constant_handler(
        self, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> bool:
        return self.can_constant_fold_through() and check_unspec_or_constant_args(
            args, kwargs
        )

    @staticmethod
    def _make_handler(
        fn: Callable[..., Any], arg_types: list[type], has_kwargs: bool
    ) -> Callable[
        [
            "InstructionTranslatorBase",
            list[VariableTracker],
            dict[str, VariableTracker],
        ],
        VariableTracker | None,
    ]:
        from .lazy import LazyConstantVariable, LazyVariableTracker

        obj = BuiltinVariable(fn)
        handlers: list[_HandlerCallback] = []

        lazy_types = [t for t in arg_types if issubclass(t, LazyVariableTracker)]
        if lazy_types:
            if not all(issubclass(t, LazyConstantVariable) for t in lazy_types):
                # Realize non-constant lazy args and re-dispatch.  Any
                # LazyConstantVariable args are kept and handled on the
                # second dispatch through the branch below.
                return lambda tx, args, kwargs: obj.call_function(
                    tx,
                    [
                        a.realize()
                        if isinstance(a, LazyVariableTracker)
                        and not isinstance(a, LazyConstantVariable)
                        else a
                        for a in args
                    ],
                    kwargs,
                )

            # Only LazyConstantVariable lazy types.  Install type guards
            # and resolve the dispatch type.  If the resolved type is
            # ConstantVariable (the common case), delegate to a handler
            # built for ConstantVariable.  Otherwise (e.g. specialize_int=
            # False turned the int into a SymNodeVariable), realize and
            # re-dispatch so the correct handler is used.
            inner_handler = BuiltinVariable._make_handler(
                fn,
                [
                    ConstantVariable if issubclass(t, LazyConstantVariable) else t
                    for t in arg_types
                ],
                has_kwargs,
            )

            def lazy_constant_handler(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                for a in args:
                    if isinstance(a, LazyConstantVariable):
                        if a.get_handler_type_for_dispatch() is not ConstantVariable:
                            return obj.call_function(
                                tx,
                                [
                                    v.realize()
                                    if isinstance(v, LazyConstantVariable)
                                    else v
                                    for v in args
                                ],
                                kwargs,
                            )
                return inner_handler(tx, args, kwargs)

            return lazy_constant_handler

        if inspect.isclass(fn) and (
            issubclass(fn, BaseException)
            # GeneratorExit doesn't inherit from Exception
            # >>> issubclass(GeneratorExit, Exception)
            # False
            or fn is GeneratorExit
        ):

            def create_exception_class_object(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker:
                if fn is AssertionError and not all(
                    x.is_python_constant() and isinstance(x.as_python_constant(), str)
                    for x in args
                ):
                    unimplemented(
                        gb_type="assert with non-string message",
                        context=str(args),
                        explanation="Dynamo only supports asserts with string messages",
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )

                return variables.ExceptionVariable(fn, args, kwargs)

            return create_exception_class_object

        if obj.can_insert_in_graph() and not (
            fn is operator.getitem
            and not issubclass(arg_types[0], variables.TensorVariable)
        ):
            if obj.tensor_args_type(arg_types):
                return obj._handle_insert_op_in_graph
            elif has_kwargs:
                # need runtime check for kwargs
                handlers.append(obj._handle_insert_op_in_graph)

        # Handle binary ops (e.g. __add__ / __radd__, __iadd__, etc.)
        # NB: Tensor args are handled above and not here
        if len(arg_types) == 2 and not has_kwargs:
            # Try to find a handler for the arg types; otherwise, fall through to constant handler
            binop_handlers = BuiltinVariable._find_binop_handler(fn, *arg_types)
            if not binop_handlers:
                pass
            elif len(binop_handlers) == 1:
                (binop_handler,) = binop_handlers
                handlers.append(lambda tx, args, _: binop_handler(tx, *args))
            else:

                def call_binop_handlers(
                    tx: "InstructionTranslatorBase", args: Any, _: Any
                ) -> Any:
                    # pyrefly: ignore [not-iterable]
                    for fn in binop_handlers:
                        rv = fn(tx, *args)
                        if rv:
                            return rv
                    return None

                handlers.append(call_binop_handlers)

        self_handler = getattr(obj, f"call_{fn.__name__}", None)
        if self_handler:

            def call_self_handler(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                try:
                    # pyrefly: ignore [not-callable]
                    return self_handler(tx, *args, **kwargs)
                except TypeError:
                    # Check if binding is bad. inspect signature bind is expensive.
                    # So check only when handler call fails.
                    try:
                        # pyrefly: ignore [bad-argument-type]
                        inspect.signature(self_handler).bind(tx, *args, **kwargs)
                    except TypeError as e:
                        has_constant_handler = obj.has_constant_handler(args, kwargs)
                        if not has_constant_handler:
                            log.warning(
                                "incorrect arg count %s %s and no constant handler",
                                self_handler,
                                e,
                            )
                            unimplemented(
                                gb_type="invalid call to builtin op handler",
                                context=f"invalid args to {self_handler}: {args} {kwargs}",
                                explanation=f"Encountered TypeError when trying to handle op {fn.__name__}",
                                hints=[*graph_break_hints.DIFFICULT],
                            )
                    else:
                        raise
                except Unsupported as exc:
                    has_constant_handler = obj.has_constant_handler(args, kwargs)
                    if not has_constant_handler:
                        raise
                    # Actually, we will handle this just fine
                    exc.remove_from_stats()
                return None

            handlers.append(call_self_handler)

        if obj.can_constant_fold_through():
            if (
                all(issubclass(x, ConstantVariable) for x in arg_types)
                and not has_kwargs
            ):

                def constant_fold_handler(
                    tx: "InstructionTranslatorBase",
                    args: list[VariableTracker],
                    kwargs: dict[str, VariableTracker],
                ) -> VariableTracker | None:
                    # fast path
                    try:
                        res = fn(
                            *[x.as_python_constant() for x in args],
                        )
                    except Exception as exc:
                        raise_observed_exception(
                            type(exc),
                            tx,
                            args=list(exc.args),
                        )
                    except AsPythonConstantNotImplementedError as exc:
                        unimplemented(
                            gb_type="constant fold exception",
                            context=f"attempted to run function {fn} with arguments {args}",
                            explanation="Encountered exception when attempting to constant fold.",
                            hints=[*graph_break_hints.DYNAMO_BUG],
                            from_exc=exc,
                        )
                    return VariableTracker.build(tx, res)

            else:

                def constant_fold_handler(
                    tx: "InstructionTranslatorBase",
                    args: list[VariableTracker],
                    kwargs: dict[str, VariableTracker],
                ) -> VariableTracker | None:
                    # path with a runtime check
                    if check_unspec_or_constant_args(args, kwargs):
                        try:
                            res = fn(
                                *[x.as_python_constant() for x in args],
                                **{
                                    k: v.as_python_constant() for k, v in kwargs.items()
                                },
                            )
                        except AsPythonConstantNotImplementedError as exc:
                            unimplemented(
                                gb_type="constant fold exception",
                                context=f"attempted to run function {fn} with arguments {args}",
                                explanation="Encountered exception when attempting to constant fold.",
                                hints=[*graph_break_hints.DYNAMO_BUG],
                                from_exc=exc,
                            )
                        except Exception as exc:
                            raise_observed_exception(
                                type(exc),
                                tx,
                                args=list(exc.args),
                            )
                        return VariableTracker.build(tx, res)
                    return None

            handlers.append(constant_fold_handler)

        def call_unimplemented(args: list[VariableTracker]) -> None:
            real_arg_types = [arg.python_type_name() for arg in args]
            unimplemented(
                gb_type="Failed to trace builtin operator",
                context=f"builtin {fn.__name__} {arg_types} {has_kwargs}",
                explanation=f"Dynamo does not know how to trace builtin operator `{fn.__name__}` "
                f"with argument types {real_arg_types} (has_kwargs {has_kwargs})",
                hints=[
                    f"Avoid calling builtin `{fn.__name__}` with argument types {real_arg_types}. "
                    f"Consider using an equivalent alternative function/method to `{fn.__name__}`.",
                    "If you are attempting to call a logging function (e.g. `print`), "
                    "you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.",
                    "Please report an issue to PyTorch.",
                ],
            )

        if len(handlers) == 0:
            return lambda tx, args, kwargs: call_unimplemented(args)
        elif len(handlers) == 1:
            (handler,) = handlers

            def builtin_dispatch(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                rv = handler(tx, args, kwargs)
                if rv:
                    return rv
                call_unimplemented(args)
                return rv

        else:

            def builtin_dispatch(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                rv = None
                for fn in handlers:
                    rv = fn(tx, args, kwargs)
                    if rv:
                        return rv
                call_unimplemented(args)
                return rv

        return builtin_dispatch

    @staticmethod
    def _constant_eval_numeric_expr(node: ast.AST) -> bool:
        allowed_nodes = (
            ast.Expression,
            ast.Constant,
            ast.UnaryOp,
            ast.BinOp,
            ast.UAdd,
            ast.USub,
            ast.Invert,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
        )
        return all(
            isinstance(child, allowed_nodes)
            and (
                not isinstance(child, ast.Constant)
                or isinstance(child.value, (bool, int, float, complex))
            )
            for child in ast.walk(node)
        )

    @staticmethod
    def _constant_eval_result(
        tx: "InstructionTranslatorBase", tree: ast.Expression, filename: str
    ) -> VariableTracker | None:
        if any(isinstance(child, ast.Call) for child in ast.walk(tree)):
            return None
        ast.fix_missing_locations(tree)
        try:
            result = ast.literal_eval(tree)
        except ValueError:
            if not BuiltinVariable._constant_eval_numeric_expr(tree):
                return None
            try:
                result = eval(
                    compile(tree, filename, "eval"),
                    {"__builtins__": {}},
                    {},
                )
            except Exception as exc:
                raise_observed_exception(type(exc), tx, args=list(exc.args))
        return VariableTracker.build(tx, result)

    def call_eval(
        self,
        tx: "InstructionTranslatorBase",
        source: VariableTracker,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker | None:
        if args or kwargs:
            return None
        if not source.is_python_constant():
            return None
        source_str = source.as_python_constant()
        if not isinstance(source_str, str):
            return None

        try:
            tree = ast.parse(source_str.strip(), mode="eval")
        except SyntaxError as exc:
            raise_observed_exception(SyntaxError, tx, args=[exc.msg])

        return self._constant_eval_result(tx, tree, "<torch._dynamo.eval>")

    def call_vars(self, tx: "InstructionTranslatorBase", *args: Any) -> VariableTracker:
        if len(args) == 0:
            return self._call_frame_locals_snapshot(tx)
        if len(args) != 1:
            raise AssertionError(f"vars() expected 1 argument, got {len(args)}")
        # vars(obj) is obj.__dict__ if __dict__ is present else TypeError
        try:
            return args[0].var_getattr(tx, "__dict__")
        except ObservedAttributeError:
            raise_observed_exception(TypeError, tx)

    def call_locals(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        if len(args) != 0:
            raise_observed_exception(TypeError, tx)
        return self._call_frame_locals_snapshot(tx)

    @staticmethod
    def _call_frame_locals_snapshot(tx: "InstructionTranslatorBase") -> VariableTracker:
        frame_local_names = set(tx.f_code.co_varnames) | set(tx.cell_and_freevars())
        cell_and_freevars = set(tx.cell_and_freevars())
        frame_locals = {}
        for name, value in tx.symbolic_locals.items():
            if name not in frame_local_names:
                continue
            if name in cell_and_freevars:
                value = tx.output.side_effects.load_cell(value)
            if type.__instancecheck__(NullVariable, value) or isinstance(
                value, variables.DeletedVariable
            ):
                continue
            frame_locals[ConstantVariable.create(name)] = value
        return ConstDictVariable(
            frame_locals,
            dict,
            mutation_type=ValueMutationNew(),
        )

    def _handle_insert_op_in_graph(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls

        if kwargs and not self.tensor_args(*args, *kwargs.values()):
            return None

        # insert handling for torch function here
        from .builder import SourcelessBuilder
        from .torch_function import can_dispatch_torch_function, dispatch_torch_function

        global BUILTIN_TO_TENSOR_RFN_MAP, BUILTIN_TO_TENSOR_FN_MAP
        if can_dispatch_torch_function(tx, args, kwargs):
            # Only remap the fn to tensor methods if we aren't exporting
            # export serde does not handle method descriptors today
            if not tx.export:
                # Ensure the builtin maps are populated before accessing them
                populate_builtin_to_tensor_fn_map()
                # Use sourceless builder, we built the map ourselves
                if not args[0].is_tensor():
                    if self.fn in BUILTIN_TO_TENSOR_RFN_MAP:
                        func = BUILTIN_TO_TENSOR_RFN_MAP[self.fn]
                    else:
                        func = BUILTIN_TO_TENSOR_FN_MAP[self.fn]

                    tmp = args[0]
                    # swap args and call reverse version of func
                    args[0] = args[1]  # type: ignore[index]
                    args[1] = tmp  # type: ignore[index]
                else:
                    func = BUILTIN_TO_TENSOR_FN_MAP[self.fn]
            else:
                func = self.fn

            fn_var = SourcelessBuilder.create(tx, func)

            return dispatch_torch_function(tx, fn_var, args, kwargs)

        fn = self.fn
        try:
            # Constant fold for constant tensor and python constants
            if self.python_and_tensor_constant_only(*args, **kwargs):
                from ..bytecode_transformation import unique_id
                from .functions import invoke_and_store_as_constant

                return invoke_and_store_as_constant(
                    tx, fn, unique_id(fn.__name__), args, kwargs
                )

            if fn in IN_PLACE_DESUGARING_MAP and isinstance(
                args[0], variables.ConstantVariable
            ):
                # In-place operators like += usually mustate tensor
                # values, but in the edge case of immutable values they
                # re-bind the variable.
                #
                # The easiest way to keep the graph consistent in this
                # scenario is to de-sugar eagerly.
                fn = IN_PLACE_DESUGARING_MAP[fn]
                args = [args[0], args[1]]  # type: ignore[assignment]

            if fn is operator.getitem and isinstance(args[1], SymNodeVariable):
                # Standard indexing will force specialization due to
                # __index__.  Rewrite as a regular torch op which will
                # trace fine
                fn = torch.select
                args = [
                    args[0],
                    variables.VariableTracker.build(tx, 0),
                    args[1],
                ]  # type: ignore[assignment]

            # Interaction between ndarray and tensors:
            #   We prefer the tensor op whenever there are tensors involved
            # NB: Use exact type check here - NumpyNdarrayVariable is a TensorVariable
            # subclass but should NOT trigger the tensor path
            if check_numpy_ndarray_args(args, kwargs) and not any(
                type(arg) is TensorVariable for arg in args
            ):
                proxy = tx.output.create_proxy(
                    "call_function",
                    numpy_operator_wrapper(fn),
                    *proxy_args_kwargs(args, kwargs),
                )

                return wrap_fx_proxy_cls(variables.NumpyNdarrayVariable, tx, proxy)

            if (
                fn in (operator.eq, operator.ne)
                and len(args) == 2
                and args[0].is_tensor()
            ):
                # Dynamo expects `__eq__` / `__ne__` strings while operator.{eq,ne}
                # provides call_function dispatch first.
                method_name = "__eq__" if fn is operator.eq else "__ne__"
                return args[0].call_method(tx, method_name, list(args[1:]), kwargs)
            proxy = tx.output.create_proxy(
                "call_function",
                fn,
                *proxy_args_kwargs(args, kwargs),
            )
            if any(isinstance(arg, FakeItemVariable) for arg in args):
                return wrap_fx_proxy_cls(
                    FakeItemVariable,
                    tx,
                    proxy,
                )
            elif check_unspec_python_args(args, kwargs):
                _args, _kwargs = self.unwrap_unspec_args_kwargs(args, kwargs)
                raw_value = fn(*_args, **_kwargs)

                need_unwrap = any(
                    x.need_unwrap
                    for x in itertools.chain(args, kwargs.values())
                    if isinstance(x, variables.UnspecializedPythonVariable)
                )

                return wrap_fx_proxy_cls(
                    UnspecializedPythonVariable,
                    tx,
                    proxy,
                    raw_value=raw_value,
                    need_unwrap=need_unwrap,
                )
            elif all(isinstance(x, SymNodeVariable) for x in args):
                return SymNodeVariable.create(tx, proxy, None)
            else:
                # Work around for vision_maskrcnn due to precision difference
                # specialize the dividend when float divide by tensor
                if fn is operator.truediv and isinstance(
                    args[0], variables.UnspecializedPythonVariable
                ):
                    args = list(args)
                    args[0] = args[0].as_python_constant()
                return wrap_fx_proxy(tx, proxy)

        except NotImplementedError:
            unimplemented(
                gb_type="unimplemented builtin op on tensor arguments",
                context=f"partial tensor op: {self} {args} {kwargs}",
                explanation=f"Dynamo does not know how to trace builtin operator {self.fn} with tensor arguments",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

    call_function_handler_cache: dict[
        tuple[object, ...],
        Callable[
            [
                "InstructionTranslatorBase",
                list[VariableTracker],
                dict[str, VariableTracker],
            ],
            VariableTracker,
        ],
    ] = {}

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        key: tuple[object, ...]
        if kwargs:
            kwargs = {k: v.realize() for k, v in kwargs.items()}
            key = (self.fn, *(type(x) for x in args), True)
        else:
            key = (self.fn, *(type(x) for x in args))

        handler = self.call_function_handler_cache.get(key)
        if not handler:
            self.call_function_handler_cache[key] = handler = self._make_handler(  # type: ignore[assignment]
                self.fn, [type(x) for x in args], bool(kwargs)
            )
        if handler is None:
            raise AssertionError(
                f"No handler found for {self.fn} with args {[type(x) for x in args]}"
            )
        return handler(tx, args, kwargs)  # type: ignore[return-value]

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if self.fn is object and name == "__setattr__":
            if len(args) != 3:
                raise AssertionError(
                    f"object.__setattr__ expects 3 args, got {len(args)}"
                )
            if len(kwargs) != 0:
                raise AssertionError(
                    f"object.__setattr__ expects no kwargs, got {len(kwargs)}"
                )
            obj, name_var, val = args
            obj = obj.realize()
            if (
                isinstance(obj, UserDefinedObjectVariable)
                and tx.output.side_effects.is_attribute_mutation(obj)
                and name_var.is_python_constant()
            ):
                return obj.method_setattr_standard(tx, name_var, val)

        if name == "__new__":
            # Supported __new__ methods
            if self.fn is object and len(args) == 1:
                if len(kwargs) != 0:
                    raise AssertionError(
                        f"object.__new__ expects no kwargs, got {len(kwargs)}"
                    )
                return tx.output.side_effects.track_new_user_defined_object(
                    self,
                    args[0],
                    args[1:],
                    tx=tx,
                )

            if self.fn is tuple and len(args) == 2 and not kwargs:
                if isinstance(args[0], BuiltinVariable) and args[0].fn is tuple:
                    init_args = unpack_iterable(tx, args[1])
                    return variables.TupleVariable(
                        init_args, mutation_type=ValueMutationNew()
                    )

                return tx.output.side_effects.track_new_user_defined_object(
                    self,
                    args[0],
                    args[1:],
                    tx=tx,
                )

        if name in _BUILTIN_CONSTANT_FOLDABLE_METHODS.get(self.fn, ()):
            if all(a.is_python_constant() for a in args) and all(
                v.is_python_constant() for v in kwargs.values()
            ):
                try:
                    fn = getattr(self.fn, name)
                    res = fn(
                        *(a.as_python_constant() for a in args),
                        **{k: v.as_python_constant() for k, v in kwargs.items()},
                    )
                    return VariableTracker.build(tx, res)
                except Exception as e:
                    raise_observed_exception(
                        type(e),
                        tx,
                        args=list(e.args),
                    )

        if self.fn is object and name == "__init__":
            # object.__init__ is a no-op
            return variables.ConstantVariable.create(None)

        if self.fn in (set, frozenset, list, tuple):
            if isinstance(args[0], variables.UserDefinedObjectVariable):
                if args[0]._base_vt is None:
                    raise AssertionError(
                        "UserDefinedObjectVariable._base_vt must not be None"
                    )
                return args[0]._base_vt.call_method(tx, name, args[1:], kwargs)
            else:
                return args[0].call_method(tx, name, args[1:], kwargs)

        if self.fn is str and len(args) >= 1:
            resolved_fn = getattr(self.fn, name, None)
            if resolved_fn in str_methods:
                # Only delegate to ConstantVariable, not other types that happen to be constants
                if isinstance(args[0], ConstantVariable):
                    return args[0].call_method(tx, name, args[1:], kwargs)

        if self.fn is float and len(args) >= 1:
            # Only delegate to ConstantVariable, not other types that happen to be constants
            if isinstance(args[0], ConstantVariable):
                return VariableTracker.build(
                    tx, getattr(float, name)(args[0].as_python_constant())
                )

        if name == "__len__" and len(args) == 1 and not kwargs:
            # type.__len__(instance) → len(instance)
            # e.g. list.__len__(my_list) → len(my_list)
            return generic_len(tx, args[0])

        if name == "__repr__" and len(args) == 1 and not kwargs:
            return super().call_method(tx, name, args, kwargs)

        if name == "__iter__" and len(args) == 1 and not kwargs:
            # type.__iter__(instance) → iter(instance)
            # e.g., tuple.__iter__(my_tuple) → iter(my_tuple)
            # For builtin types called on user-defined subclasses, use the base iterator
            return generic_getiter(tx, args[0])

        if name == "__neg__" and len(args) == 1 and not kwargs:
            # type.__neg__(instance) → neg(instance)
            # e.g., int.__neg__(4) → neg(4)
            return generic_neg(tx, args[0])

        if name == "__pos__" and len(args) == 1 and not kwargs:
            # type.__pos__(instance) → pos(instance)
            # e.g., int.__pos__(4) → pos(4)
            return generic_pos(tx, args[0])

        if name == "__abs__" and len(args) == 1 and not kwargs:
            # type.__abs__(instance) → abs(instance)
            # e.g., int.__abs__(-4) → abs(-4)
            return generic_abs(tx, args[0])

        return super().call_method(tx, name, args, kwargs)

    def call_int(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        return generic_int(tx, arg)

    def call_float(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        return generic_float(tx, arg)

    def call_bool(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        # Emulate PyBool_Type.tp_vectorcall which boils down to PyObject_IsTrue.
        return generic_bool(tx, arg)

    def call_hash(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        from .object_protocol import generic_hash

        return generic_hash(tx, arg)

    def call_repr(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        return generic_repr(tx, arg)

    def call_str(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        if isinstance(
            arg,
            (variables.ExceptionVariable, variables.UserDefinedExceptionObjectVariable),
        ):
            if len(arg.args) == 0:
                return VariableTracker.build(tx, "")
            elif len(arg.args) == 1:
                return BuiltinVariable(str).call_function(tx, [arg.args[0]], {})
            else:
                tuple_var = variables.TupleVariable(list(arg.args))
                return BuiltinVariable(str).call_function(tx, [tuple_var], {})

        # Handle `str` on a user defined function or object
        if isinstance(arg, (variables.UserFunctionVariable)):
            return VariableTracker.build(tx, str(arg.fn))
        elif isinstance(arg, (variables.UserDefinedObjectVariable)):
            # Check if object has __str__ method
            if hasattr(arg.value, "__str__"):
                str_method = arg.value.__str__
            elif hasattr(arg.value, "__repr__"):
                # account for __repr__ functions when __str__ is absent
                str_method = arg.value.__repr__
            else:
                unimplemented(
                    gb_type="failed to call str() on user defined object",
                    context=str(arg),
                    explanation="User defined object has no __str__ or __repr__ method",
                    hints=[*graph_break_hints.USER_ERROR],
                )

            if type(arg.value).__str__ is object.__str__:
                # Rely on the object str method
                try:
                    # pyrefly: ignore [unbound-name]
                    return VariableTracker.build(tx, str_method())
                except AttributeError:
                    # Graph break
                    return None
            elif is_wrapper_or_member_descriptor(str_method):
                unimplemented(
                    gb_type="Attempted to a str() method implemented in C/C++",
                    context="",
                    explanation=f"{type(arg.value)} has a C/C++ based str method. This is not supported.",
                    hints=["Write the str method in Python"],
                )
            else:
                # Overrides for custom str method
                # Pass method as function to call tx.inline_user_function_return
                bound_method = str_method.__func__  # type: ignore[attr-defined]

                try:
                    # Only supports certain function types
                    user_func_variable = VariableTracker.build(tx, bound_method)
                except AssertionError:
                    # Won't be able to do inline the str method, return to avoid graph break
                    log.warning("Failed to create UserFunctionVariable", exc_info=True)
                    return None

                # Inline the user function
                return user_func_variable.call_function(tx, [arg], {})
        return None

    def call___build_class__(self, tx, *args, **kwargs):
        def fail(args, kwargs) -> NoReturn:
            unimplemented(
                gb_type="Invalid call to __build_class__",
                context=f"Non-constant args to __build_class__: {args} {kwargs}",
                explanation="Cannot trace class definition: the class body function is unsupported or the base class argument are not compile-time constants",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

        if not torch._dynamo.config.enable_trace_load_build_class:
            fail(args, kwargs)

        try:
            if isinstance(args[0], variables.NestedUserFunctionVariable):
                fn = args[0].get_function(allow_sourced_cells=True)
            else:
                fn = args[0].get_function()
        except NotImplementedError:
            fail(args, kwargs)

        if check_constant_args(args[1:], kwargs):
            r = builtins.__build_class__(
                fn,  # type: ignore[possibly-undefined]
                *[a.as_python_constant() for a in args[1:]],
            )
            return VariableTracker.build(tx, r)
        else:
            fail(args, kwargs)

    def _call_min_max(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker | None:
        if len(args) == 1:
            items = unpack_iterable(tx, args[0])
            return self._call_min_max_seq(tx, items)
        elif len(args) == 2:
            return self._call_min_max_binary(tx, args[0], args[1])
        elif len(args) > 2:
            return self._call_min_max_seq(tx, list(args))
        return None

    def _call_min_max_seq(
        self, tx: "InstructionTranslatorBase", items: list[VariableTracker]
    ) -> VariableTracker:
        if len(items) <= 0:
            raise AssertionError("_call_min_max_seq requires at least one item")
        if len(items) == 1:
            return items[0]

        return functools.reduce(functools.partial(self._call_min_max_binary, tx), items)  # type: ignore[arg-type,return-value]

    def _call_min_max_binary(
        self,
        tx: "InstructionTranslatorBase",
        a: VariableTracker | None,
        b: VariableTracker | None,
    ) -> VariableTracker | None:
        if a is None or b is None:
            # a or b could be none if we reduce and _call_min_max_binary failed
            # to return something
            return None
        if self.tensor_args(a, b):
            if not a.is_tensor():
                a, b = b, a
            if not a.is_tensor():
                raise AssertionError(
                    "Expected at least one tensor argument for min/max"
                )

            # result of an item call is a scalar convert to a tensor
            if isinstance(a, FakeItemVariable):
                a = variables.TorchInGraphFunctionVariable(torch.tensor).call_function(
                    tx, [a], {}
                )

            # Dynamic input does not get resolved, rather, gets stored as call_function
            if isinstance(a, SymNodeVariable) or isinstance(b, SymNodeVariable):
                from .builder import wrap_fx_proxy_cls

                return wrap_fx_proxy_cls(
                    type(a),
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        self.fn,
                        *proxy_args_kwargs([a, b], {}),
                    ),
                )

            # convert min/max to torch ops
            if b.is_python_constant():
                fn: VariableTracker
                if isinstance(a, variables.NumpyNdarrayVariable):
                    import numpy as np

                    fn = variables.NumpyVariable(np.clip)
                else:
                    fn = variables.TorchInGraphFunctionVariable(torch.clamp)
                kwargs = {"min": b} if (self.fn is max) else {"max": b}
                result = fn.call_function(tx, [a], kwargs)
            else:
                if isinstance(a, variables.NumpyNdarrayVariable):
                    import numpy as np

                    np_fn = {max: np.maximum, min: np.minimum}[self.fn]
                    fn = variables.NumpyVariable(np_fn)
                else:
                    torch_fn = {max: torch.maximum, min: torch.minimum}[self.fn]
                    fn = variables.TorchInGraphFunctionVariable(torch_fn)
                result = fn.call_function(tx, [a, b], {})

            # return unspec if both a, b are unspec or const
            if all(
                isinstance(
                    i,
                    (
                        variables.UnspecializedPythonVariable,
                        variables.ConstantVariable,
                    ),
                )
                for i in [a, b]
            ):
                if any(isinstance(val, FakeItemVariable) for val in [a, b]):
                    # type: ignore[arg-type]
                    return variables.FakeItemVariable.from_tensor_variable(result)

                if b.is_python_constant():
                    raw_b = b.as_python_constant()
                else:
                    raw_b = b.raw_value  # type: ignore[attr-defined]
                if self.fn is max:
                    raw_res = max(a.raw_value, raw_b)  # type: ignore[attr-defined]
                else:
                    raw_res = min(a.raw_value, raw_b)  # type: ignore[attr-defined]

                need_unwrap = any(
                    x.need_unwrap
                    for x in [a, b]
                    if isinstance(x, variables.UnspecializedPythonVariable)
                )
                return variables.UnspecializedPythonVariable.from_tensor_variable(
                    result,  # type: ignore[arg-type]
                    raw_res,
                    need_unwrap,
                )
            # otherwise return tensor
            else:
                return result
        elif isinstance(a, SymNodeVariable) or isinstance(b, SymNodeVariable):
            py_fn = torch.sym_max if self.fn is max else torch.sym_min
            proxy = tx.output.create_proxy(
                "call_function", py_fn, *proxy_args_kwargs([a, b], {})
            )
            return SymNodeVariable.create(tx, proxy, None)
        elif isinstance(a, ConstantVariable) and isinstance(b, ConstantVariable):
            value = self.fn(
                a.as_python_constant(),
                b.as_python_constant(),
            )
            return VariableTracker.build(tx, value)
        return None

    call_min = _call_min_max
    call_max = _call_min_max

    def call_abs(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        return generic_abs(tx, arg)

    def call_pos(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        return generic_pos(tx, arg)

    def call_index(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        # Specialize SymNodeVariable to a constant first, matching CPython's
        # PyNumber_Index which forces a concrete int.
        arg = specialize_symnode(arg)
        return arg.nb_index_impl(tx)

    def call_round(
        self,
        tx: "InstructionTranslatorBase",
        arg: VariableTracker,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        from .builder import SourcelessBuilder

        # Call arg.__round__()
        round_method = SourcelessBuilder.create(tx, getattr).call_function(
            tx, [arg, VariableTracker.build(tx, "__round__")], {}
        )
        return round_method.call_function(tx, list(args), kwargs)

    def call_range(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker | None:
        if check_unspec_or_constant_args(args, {}):
            return variables.RangeVariable(list(args))
        elif self._dynamic_args(*args):
            args = tuple(VariableTracker.build(tx, guard_if_dyn(arg)) for arg in args)
            return variables.RangeVariable(list(args))
        # None no-ops this handler and lets the driving function proceed
        return None

    def _dynamic_args(self, *args: VariableTracker, **kwargs: VariableTracker) -> bool:
        return any(isinstance(x, SymNodeVariable) for x in args) or any(
            isinstance(x, SymNodeVariable) for x in kwargs.values()
        )

    def call_slice(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        if not 1 <= len(args) < 4:
            raise_type_error(tx, f"slice expected at least 1 argument, got {len(args)}")
        return variables.SliceVariable(list(args), tx)

    def _dyn_proxy(
        self, tx: "InstructionTranslatorBase", *args: Any, **kwargs: Any
    ) -> VariableTracker:
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_function", self.fn, *proxy_args_kwargs(args, kwargs)
            ),
        )

    def call_tuple(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker | None:
        # ref: https://github.com/python/cpython/blob/main/Objects/abstract.c#L2004-L2078
        if kwargs:
            raise_type_error(
                tx,
                f"{self.fn.__name__}() takes no keyword arguments",
            )
        if len(args) == 0:
            return TupleVariable([], mutation_type=ValueMutationNew())
        elif len(args) > 1:
            raise_type_error(
                tx,
                f"{self.fn.__name__} expected at most 1 argument, got {len(args)}",
            )

        obj = args[0]
        if isinstance(obj, TupleVariable) and obj.python_type() is tuple:
            return obj

        items = unpack_iterable(tx, args[0])
        return TupleVariable(items, mutation_type=ValueMutationNew())

    def call_callable(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        from .functions import BaseUserFunctionVariable, FunctoolsPartialVariable
        from .nn_module import NNModuleVariable

        if isinstance(
            arg,
            (
                variables.UserDefinedClassVariable,
                BaseUserFunctionVariable,
                FunctoolsPartialVariable,
                NNModuleVariable,
            ),
        ):
            return variables.ConstantVariable.create(True)
        elif isinstance(arg, UserDefinedVariable):
            return VariableTracker.build(tx, callable(arg.value))
        elif isinstance(
            arg,
            (
                ConstantVariable,
                SymNodeVariable,
                StringFormatVariable,
                TensorVariable,
                ListVariable,
                TupleVariable,
                ListIteratorVariable,
            ),
        ):
            return variables.ConstantVariable.create(False)
        else:
            return None

    def call_cast(
        self, _: Any, *args: VariableTracker, **kwargs: VariableTracker
    ) -> VariableTracker | None:
        if len(args) == 2:
            return args[1]

        unimplemented(
            gb_type="bad args to builtin cast()",
            context=f"got args {args} {kwargs}",
            explanation="Dynamo expects exactly 2 args to builtin cast().",
            hints=["Ensure your call to cast() has exactly 2 arguments."],
        )

    def call_dir(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        if isinstance(arg, variables.UserDefinedClassVariable):
            return VariableTracker.build(tx, dir(arg.value))
        if isinstance(arg, BuiltinVariable):
            return VariableTracker.build(tx, dir(arg.fn))
        # Enable specialized VTs for constants to work with dir()
        if arg.is_python_constant():
            return VariableTracker.build(tx, dir(arg.as_python_constant()))
        return None

    def call_set(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/main/Objects/setobject.c#L2708-L2735
        if len(args) == 0:
            return variables.SetVariable(set(), mutation_type=ValueMutationNew())
        elif len(args) > 1:
            raise_type_error(
                tx,
                f"set expected at most 1 argument, got {len(args)}",
            )
        elif kwargs:
            raise_type_error(
                tx,
                "set() takes no keyword arguments",
            )

        s = SetVariable([], mutation_type=ValueMutationNew())
        s.call_method(tx, "update", [args[0]], kwargs)
        return s

    def call_frozenset(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        if len(args) == 0:
            return variables.FrozensetVariable(set(), mutation_type=ValueMutationNew())
        elif len(args) > 1:
            raise_type_error(
                tx,
                f"frozenset expected at most 1 argument, got {len(args)}",
            )
        elif kwargs:
            raise_type_error(
                tx,
                "frozenset() takes no keyword arguments",
            )

        if istype(args[0], variables.FrozensetVariable):
            # CPython: frozenset(existing_frozenset) returns the same object.
            return args[0]

        items = unpack_iterable(tx, args[0])
        fs = FrozensetVariable(items, mutation_type=ValueMutationNew())
        return fs

    def call_zip(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.3/Python/bltinmodule.c#L2822-L2887
        if kwargs:
            if not (len(kwargs) == 1 and "strict" in kwargs):
                raise_args_mismatch(
                    tx,
                    "zip",
                    "1 kwargs (`strict`)",
                    f"{len(kwargs)} kwargs",
                )
        strict = kwargs.pop("strict", ConstantVariable.create(False))
        items = []
        for arg in args:
            items.append(generic_getiter(tx, arg))
        iter_args = TupleVariable(items, mutation_type=ValueMutationNew())
        return variables.ZipVariable(
            iter_args,
            strict=strict.as_python_constant(),
            mutation_type=ValueMutationNew(),
        )

    def call_len(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        return generic_len(tx, args[0])

    def call_getitem(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        return vt_getitem(tx, args[0], args[1])

    def call_isinstance(
        self,
        tx: "InstructionTranslatorBase",
        arg: VariableTracker,
        isinstance_type_var: VariableTracker,
    ) -> VariableTracker:
        try:
            arg_type = arg.python_type()
        except NotImplementedError:
            unimplemented(
                gb_type="builtin isinstance() cannot determine type of argument",
                context=f"isinstance({arg}, {isinstance_type_var})",
                explanation=f"Dynamo doesn't have a rule to determine the type of argument {arg}",
                hints=[*graph_break_hints.DYNAMO_BUG],
            )
        isinstance_type = isinstance_type_var.as_python_constant()
        if isinstance(arg, variables.TensorVariable) and arg.dtype is not None:

            def _tensor_isinstance(
                tensor_var: VariableTracker, tensor_type: Any
            ) -> bool:
                def check_type(ty: Any) -> bool:
                    if ty not in tensortype_to_dtype:
                        example_val = arg.as_proxy().node.meta["example_value"]
                        if (
                            is_traceable_wrapper_subclass(example_val)
                            and ty is torch.nn.parameter.Parameter
                        ):
                            # N.B: we are calling isinstance directly on the example value.
                            # torch.nn.Parameter has a meta-class that overrides __isinstance__,
                            # the isinstance check here allows us to invoke that logic.
                            return isinstance(example_val, ty)
                        else:
                            return issubclass(arg.python_type(), ty)

                    dtypes = tensortype_to_dtype[ty]
                    # pyrefly: ignore [missing-attribute]
                    return arg.dtype in dtypes

                if type(tensor_type) is tuple:
                    return any(check_type(ty) for ty in tensor_type)
                else:
                    return check_type(tensor_type)

            return VariableTracker.build(tx, _tensor_isinstance(arg, isinstance_type))
        # UserDefinedObject with C extensions can have torch.Tensor attributes,
        # so break graph.
        if isinstance(arg, variables.UserDefinedObjectVariable) and isinstance(
            arg.value, types.MemberDescriptorType
        ):
            unimplemented(
                gb_type="isinstance() called on user defined object with C extensions",
                context=f"isinstance({arg}, {isinstance_type})",
                explanation="User-defined object with C extensions can have torch.Tensor "
                "attributes; intentionally graph breaking.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        # handle __instancecheck__ defined in user class
        if (
            isinstance(arg, variables.UserDefinedObjectVariable)
            and "__instancecheck__" in isinstance_type.__class__.__dict__
        ):
            return VariableTracker.build(
                tx,
                isinstance_type.__class__.__instancecheck__(isinstance_type, arg.value),
            )

        if isinstance(arg, variables.UserDefinedExceptionClassVariable):
            # pyrefly: ignore [unbound-name]
            return VariableTracker.build(tx, isinstance(arg_type, isinstance_type))

        isinstance_type_tuple: tuple[type, ...]
        if isinstance(isinstance_type, type) or callable(
            # E.g. isinstance(obj, typing.Sequence)
            getattr(isinstance_type, "__instancecheck__", None)
        ):
            isinstance_type_tuple = (isinstance_type,)
        elif isinstance(isinstance_type, types.UnionType):
            isinstance_type_tuple = typing.get_args(isinstance_type)
        elif isinstance(isinstance_type, tuple) and all(
            isinstance(tp, type) or callable(getattr(tp, "__instancecheck__", None))
            for tp in isinstance_type
        ):
            isinstance_type_tuple = isinstance_type
        else:
            raise_observed_exception(
                TypeError,
                tx,
                args=[
                    "isinstance() arg 2 must be a type, a tuple of types, or a union"
                ],
            )

        try:
            # NB: `isinstance()` does not call `__subclasscheck__` but use `__instancecheck__`.
            # But usually `isinstance(obj, type_info)` and `issubclass(type(obj), type_info)` gives
            # the same result.
            # WARNING: This might run arbitrary user code `__subclasscheck__` and we did not trace
            # through it. This is a limitation of the current implementation.
            # Usually `__subclasscheck__` and `__instancecheck__` can be constant fold through, it
            # might not be a big issue and we trade off it for performance.
            val = issubclass(arg_type, isinstance_type_tuple)
        except TypeError:
            val = arg_type in isinstance_type_tuple
        return VariableTracker.build(tx, val)

    def call_issubclass(
        self,
        tx: "InstructionTranslatorBase",
        left_ty: VariableTracker,
        right_ty: VariableTracker,
    ) -> VariableTracker:
        """Checks if first arg is subclass of right arg"""
        from .object_protocol import generic_issubclass

        return generic_issubclass(tx, left_ty, right_ty)

    def call_super(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker:
        return variables.SuperVariable(a, b)

    def call_next(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        arg = args[0]
        try:
            return arg.next_variable(tx)
        except ObservedUserStopIteration:
            if len(args) == 2:
                return args[1]
            raise
        except Unsupported as ex:
            if isinstance(arg, variables.BaseListVariable):
                ex.remove_from_stats()
                return arg.items[0]
            raise

    def call_map(
        self,
        tx: "InstructionTranslatorBase",
        fn: VariableTracker,
        *seqs: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        if len(seqs) == 0:
            raise_observed_exception(
                TypeError,
                tx,
                args=["map() must have at least two arguments."],
            )

        strict = ConstantVariable.create(False)
        if kwargs:
            if sys.version_info >= (3, 14):
                if not (len(kwargs) == 1 and "strict" in kwargs):
                    raise_args_mismatch(
                        tx,
                        "map",
                        "1 kwargs (`strict`)",
                        f"{len(kwargs)} kwargs",
                    )
                strict = kwargs.pop("strict", ConstantVariable.create(False))
            else:
                raise_args_mismatch(
                    tx,
                    "map",
                    "0 kwargs",
                    f"{len(kwargs)} kwargs",
                )

        iterables = [generic_getiter(tx, seq) for seq in seqs]
        iter_args = TupleVariable(iterables, mutation_type=ValueMutationNew())
        return variables.MapVariable(
            fn,
            iter_args,
            strict=strict.as_python_constant(),
            mutation_type=ValueMutationNew(),
        )

    def call_filter(
        self, tx: "InstructionTranslatorBase", fn: VariableTracker, seq: VariableTracker
    ) -> VariableTracker:
        return variables.FilterVariable(
            fn,
            generic_getiter(tx, seq),
            mutation_type=ValueMutationNew(),
        )

    def var_getattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        source = self.source and AttrSource(self.source, name)
        if name == "__name__":
            return VariableTracker.build(tx, self.fn.__name__, source)
        if self.fn is object:
            # for object, we can just directly read the attribute
            try:
                value = getattr(self.fn, name)
            except AttributeError:
                raise_observed_exception(AttributeError, tx)
            if not callable(value):
                return VariableTracker.build(tx, value, source)
        attr = getattr(self.fn, name, None)
        return variables.GetAttrVariable(
            self, name, py_type=type(attr) if attr is not None else None, source=source
        )

    def call_delattr(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        name_var: VariableTracker,
    ) -> VariableTracker:
        return obj.call_method(tx, "__delattr__", [name_var], {})

    def call_type(
        self, tx: "InstructionTranslatorBase", obj: VariableTracker
    ) -> VariableTracker:
        try:
            py_type = obj.python_type()
        except NotImplementedError as error:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                str(error),
                case_name="unknown_python_type",
            ) from None

        source = obj.source and TypeSource(obj.source)
        if (
            source is None
            and isinstance(obj, variables.UserDefinedObjectVariable)
            and obj.cls_source
        ):
            source = obj.cls_source
        if py_type is torch.Tensor:
            # In some cases torch isn't available in globals
            name = tx.output.install_global_by_id("", torch)
            source = AttrSource(GlobalSource(name), "Tensor")

        return VariableTracker.build(tx, py_type, source)

    def call_reversed(
        self, tx: "InstructionTranslatorBase", obj: VariableTracker
    ) -> VariableTracker:
        # Mirrors CPython's builtin_reversed_impl (Python/enumobject.c)
        # https://github.com/python/cpython/blob/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Objects/enumobject.c#L353-L395
        # 1. Look up __reversed__ via _PyObject_LookupSpecial. If found, call it.
        # 2. Else require PySequence_Check (sq_item). If absent, TypeError.
        # 3. Else build a reverse sequence iterator over __len__ + __getitem__.

        obj_type = maybe_get_python_type(obj)

        # Type-level __reversed__ lookup, mirrors _PyObject_LookupSpecial.
        # getattr_static skips descriptors / metaclass. CPython treats
        # `__reversed__ = None` on the type as an explicit opt-out, raising
        # TypeError even if the sequence protocol would otherwise work.
        reversed_attr = inspect.getattr_static(
            obj_type, "__reversed__", _MISSING_SENTINEL
        )
        if reversed_attr is None:
            raise_type_error(tx, f"'{obj_type.__name__}' object is not reversible")
        if reversed_attr is not _MISSING_SENTINEL:
            return obj.call_method(tx, "__reversed__", [], {})

        if not pysequence_check(obj_type):
            raise_type_error(tx, "argument to reversed() must be a sequence")

        return variables.UserFunctionVariable(
            polyfills.builtins.reversed_sequence_iterator
        ).call_function(tx, [obj], {})

    def call_sorted(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker | None:
        if not isinstance(obj, variables.TensorVariable):
            list_var = variables.ListVariable(
                unpack_iterable(tx, obj),
                mutation_type=ValueMutationNew(),
            )
            list_var.call_method(tx, "sort", [], kwargs)
            return list_var
        return None

    def call_neg(
        self, tx: "InstructionTranslatorBase", a: VariableTracker
    ) -> VariableTracker:
        return generic_neg(tx, a)

    def call_format(
        self,
        tx: "InstructionTranslatorBase",
        _format_string: VariableTracker,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        format_string = _format_string.as_python_constant()
        format_string = str(format_string)
        return StringFormatVariable.create(format_string, list(args), kwargs)

    def call_id(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        if len(args) != 1:
            raise_observed_exception(
                TypeError,
                tx,
                args=[f"id() takes exactly one argument ({len(args)} given)"],
            )
        arg = args[0]

        real_id = arg.get_id(tx)
        if real_id is not None:
            if arg.source:
                guard_type = arg.get_id_guard_type()
                if guard_type is not None:
                    install_guard(arg.source.make_guard(guard_type))
            return VariableTracker.build(tx, real_id)

        return FakeIdVariable(id(arg))

    def call_deepcopy(
        self, tx: "InstructionTranslatorBase", x: VariableTracker
    ) -> VariableTracker:
        unimplemented(
            gb_type="copy.deepcopy()",
            context=f"copy.deepcopy({x})",
            explanation="Dynamo does not support copy.deepcopy()",
            hints=[
                "Avoid calling copy.deepcopy()",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def _comparison_with_tensor(
        self,
        tx: "InstructionTranslatorBase",
        left: VariableTracker,
        right: VariableTracker,
    ) -> VariableTracker:
        from .builder import wrap_fx_proxy_cls
        from .tensor import supported_tensor_comparison_op_values

        op = self.fn

        if op in [operator.is_, operator.is_not]:
            is_result = (
                left.is_tensor()
                and right.is_tensor()
                and id(extract_fake_example_value(left.as_proxy().node))
                == id(extract_fake_example_value(right.as_proxy().node))
            )
            if op is operator.is_:
                return VariableTracker.build(tx, is_result)
            else:
                return VariableTracker.build(tx, not is_result)

        if op not in supported_tensor_comparison_op_values:
            unimplemented(
                gb_type="unsupported Tensor comparison op",
                context=f"{op.__name__}({left}, {right})",
                explanation=f"Dynamo does not support the comparison op {op.__name__} "
                f"with Tensor arguments {left}, {right}",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        if (
            isinstance(left, TensorVariable)
            and isinstance(right, TensorVariable)
            and (left.size and right.size) is not None
            and left.size != right.size
        ):
            try:
                torch.broadcast_shapes(left.size, right.size)
            except RuntimeError:
                # not broadcastable, can't be compared
                unimplemented(
                    gb_type="failed to broadcast when attempting Tensor comparison op",
                    context=f"{op.__name__}({left}, {right})",
                    explanation=f"Dynamo was unable to broad cast the arguments {left}, {right} "
                    f"when attempting to trace the comparison op {op.__name__}.",
                    hints=[*graph_break_hints.USER_ERROR],
                )
        tensor_cls = left if left.is_tensor() else right
        proxy = tx.output.create_proxy(
            "call_function", op, (left.as_proxy(), right.as_proxy()), {}
        )
        return wrap_fx_proxy_cls(
            type(tensor_cls),  # handle Ndarrays and Tensors
            tx,
            proxy,
        )

    def _comparison_with_symnode(
        self,
        tx: "InstructionTranslatorBase",
        left: VariableTracker,
        right: VariableTracker,
    ) -> VariableTracker:
        from .tensor import supported_tensor_comparison_op_values

        op = self.fn

        if op not in supported_tensor_comparison_op_values:
            unimplemented(
                gb_type="unsupported SymNode comparison op",
                context=f"{op.__name__}({left}, {right})",
                explanation=f"Dynamo does not support the comparison op {op.__name__} "
                f"with SymNode arguments {left}, {right}",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

        # SymNodes are numeric (int/float/bool). The non-SymNode operand
        # must be a type that can participate in a traced numeric comparison.
        # Anything else (classes, DataPtrVariable, etc.) is a different type
        # entirely — the comparison result is known at compile time.
        non_symnode = right if isinstance(left, SymNodeVariable) else left
        if not isinstance(
            non_symnode, (SymNodeVariable, ConstantVariable, TensorVariable)
        ):
            # pyrefly: ignore [bad-argument-type]
            return VariableTracker.build(tx, op(object(), None))

        proxy = tx.output.create_proxy(
            "call_function", op, (left.as_proxy(), right.as_proxy()), {}
        )
        return SymNodeVariable.create(
            tx,
            proxy,
            sym_num=None,
        )

    def call_xor(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # Rely on constant_handler
        if isinstance(a, ConstantVariable) and isinstance(b, ConstantVariable):
            return None
        if a.is_symnode_like() and b.is_symnode_like():
            return SymNodeVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_function", operator.xor, *proxy_args_kwargs([a, b], {})
                ),
                sym_num=None,
            )

        if isinstance(a, _SET_LIKE_OP_SUPPORT):
            return a.call_method(tx, "__xor__", [b], {})
        return None

    def call_ixor(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        if isinstance(a, _SET_LIKE_OP_SUPPORT):
            return a.call_method(tx, "__ixor__", [b], {})
        return None

    def call_mul(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return generic_multiply(tx, a, b)

    def call_imul(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return generic_inplace_multiply(tx, a, b)

    def call_sub(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return binary_op(tx, a, b, "nb_subtract", "-")

    def call_isub(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return binary_iop(tx, a, b, "nb_inplace_subtract", "nb_subtract", "-=")

    def call_add(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return vt_add(tx, a, b)

    def call_iadd(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return vt_inplace_add(tx, a, b)

    def call_and_(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # Rely on constant_handler
        if isinstance(a, ConstantVariable) and isinstance(b, ConstantVariable):
            return None
        if a.is_symnode_like() and b.is_symnode_like():
            return SymNodeVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_function", operator.and_, *proxy_args_kwargs([a, b], {})
                ),
                sym_num=None,
            )
        if isinstance(a, _SET_LIKE_OP_SUPPORT):
            return a.call_method(tx, "__and__", [b], {})
        # None no-ops this handler and lets the driving function proceed
        return None

    def call_iand(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # Rely on constant_handler
        if isinstance(a, ConstantVariable) and isinstance(b, ConstantVariable):
            return None
        if a.is_symnode_like() and b.is_symnode_like():
            # In-place bitwise ops on immutable bool/int values rebind the local.
            # Emit the out-of-place op so FX codegen does not assign to a literal.
            return SymNodeVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_function", operator.and_, *proxy_args_kwargs([a, b], {})
                ),
                sym_num=None,
            )
        if isinstance(a, _SET_LIKE_OP_SUPPORT):
            return a.call_method(tx, "__iand__", [b], {})
        return None

    def call_or_(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return binary_op(tx, a, b, "nb_or", "|")

    def call_ior(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return binary_iop(tx, a, b, "nb_inplace_or", "nb_or", "|=")

    def call_lshift(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return binary_op(tx, a, b, "nb_lshift", "<<")

    def call_ilshift(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return binary_iop(tx, a, b, "nb_inplace_lshift", "nb_lshift", "<<=")

    def call_rshift(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return binary_op(tx, a, b, "nb_rshift", ">>")

    def call_irshift(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        return binary_iop(tx, a, b, "nb_inplace_rshift", "nb_rshift", ">>=")

    def call_not_(
        self, tx: "InstructionTranslatorBase", a: VariableTracker
    ) -> VariableTracker | None:
        if isinstance(a, SymNodeVariable):
            return SymNodeVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_function", operator.not_, *proxy_args_kwargs([a], {})
                ),
                sym_num=None,
            )

        # Unwrap the underlying ConstDictVariable
        if isinstance(a, DictViewVariable):
            a = a.dv_dict
        if isinstance(a, (ListVariable, ConstDictVariable, SetVariable)):
            return VariableTracker.build(tx, len(a.items) == 0)
        if isinstance(a, UserDefinedObjectVariable):
            bool_result = self.call_bool(tx, a)
            return VariableTracker.build(tx, not bool_result.value)  # type: ignore[missing-attribute]

        return None

    def call_contains(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker:
        from .object_protocol import generic_contains

        return generic_contains(tx, a, b)

    def is_python_equal(self, other: object) -> bool:
        return isinstance(other, variables.BuiltinVariable) and self.fn is other.fn


class DictBuiltinVariable(BaseBuiltinVariable):
    """Variable tracker for the `dict` builtin constructor."""

    _fn = dict

    def __init__(self, value: type = dict, **kwargs: Any) -> None:
        if value is not dict:
            raise AssertionError(f"DictBuiltinVariable value must be dict, got {value}")
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return "DictBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return DictBuiltinVariable.call_custom_dict(tx, dict, *args, **kwargs)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__new__":
            if args:
                # dict.__new__ (tp_new) ignores extra args — only the first
                # arg (the type) matters.  Pass init_args=[] so reconstruction
                # emits base_cls.__new__(cls) without extras.
                # https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L4735-L4768
                dict_vt = ConstDictVariable({}, dict, mutation_type=ValueMutationNew())
                if isinstance(args[0], DictBuiltinVariable):
                    return dict_vt
                return tx.output.side_effects.track_new_user_defined_object(
                    self,
                    args[0],
                    [],
                    tx=tx,
                )

        if name == "fromkeys":
            return DictBuiltinVariable.call_custom_dict_fromkeys(
                tx, dict, *args, **kwargs
            )

        resolved_fn = getattr(dict, name, None)
        if resolved_fn is not None and resolved_fn in dict_methods:
            if isinstance(args[0], variables.UserDefinedDictVariable):
                if args[0]._base_vt is None:
                    raise AssertionError(
                        "UserDefinedDictVariable._base_vt must not be None for dict method dispatch"
                    )
                return args[0]._base_vt.call_method(tx, name, args[1:], kwargs)
            elif isinstance(args[0], ConstDictVariable):
                return args[0].call_method(tx, name, args[1:], kwargs)

        return super().call_method(tx, name, args, kwargs)

    @staticmethod
    def call_custom_dict(
        tx: "InstructionTranslatorBase",
        user_cls: type,
        /,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        args_list = list(args)
        return tx.inline_user_function_return(
            VariableTracker.build(tx, polyfills.construct_dict),
            [VariableTracker.build(tx, user_cls), *args_list],
            kwargs,
        )

    @staticmethod
    def call_custom_dict_fromkeys(
        tx: "InstructionTranslatorBase",
        user_cls: type,
        /,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        if user_cls not in {dict, OrderedDict, defaultdict}:
            unimplemented(
                gb_type="Unsupported dict type for fromkeys()",
                context=f"{user_cls.__name__}.fromkeys(): {args} {kwargs}",
                explanation=f"Failed to call {user_cls.__name__}.fromkeys() because "
                f"{user_cls.__name__} is not any type of dict, OrderedDict, or defaultdict",
                hints=[
                    f"Ensure {user_cls.__name__} is a type of dict, OrderedDict, or defaultdict.",
                ],
            )
        if kwargs:
            # Only `OrderedDict.fromkeys` accepts `value` passed by keyword
            if (
                user_cls is not OrderedDict
                or len(args) != 1
                or len(kwargs) != 1
                or "value" not in kwargs
            ):
                raise_args_mismatch(
                    tx,
                    f"{user_cls.__name__}.fromkeys",
                    "1 args and 1 kwargs (`value`)",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            args = (*args, kwargs.pop("value"))
        if len(args) == 0:
            raise_args_mismatch(
                tx,
                f"{user_cls.__name__}.fromkeys",
                "at least 1 args",
                f"{len(args)} args",
            )
        if len(args) == 1:
            args = (*args, ConstantVariable.create(None))
        if len(args) != 2:
            raise_args_mismatch(
                tx,
                f"{user_cls.__name__}.fromkeys",
                "2 args",
                f"{len(args)} args",
            )

        arg, value = args

        def _make_result(
            items: dict[VariableTracker, VariableTracker],
        ) -> VariableTracker:
            if user_cls is OrderedDict:
                from .builder import SourcelessBuilder
                from .user_defined import OrderedDictVariable

                result = tx.output.side_effects.track_new_user_defined_object(
                    SourcelessBuilder.create(tx, dict),
                    SourcelessBuilder.create(tx, OrderedDict),
                    [],
                    tx=tx,
                )
                if not isinstance(result, OrderedDictVariable):
                    raise AssertionError(
                        f"Expected OrderedDictVariable, got {type(result)}"
                    )
                result._base_vt = ConstDictVariable(
                    items,
                    user_cls=OrderedDict,
                    mutation_type=ValueMutationNew(),
                )
                return result
            elif user_cls is defaultdict:
                from .builder import SourcelessBuilder
                from .user_defined import DefaultDictVariable

                result = tx.output.side_effects.track_new_user_defined_object(
                    SourcelessBuilder.create(tx, dict),
                    SourcelessBuilder.create(tx, defaultdict),
                    [],
                    tx=tx,
                )
                if not isinstance(result, DefaultDictVariable):
                    raise AssertionError(
                        f"Expected DefaultDictVariable, got {type(result)}"
                    )
                result._base_vt = ConstDictVariable(
                    items, mutation_type=ValueMutationNew()
                )
                return result
            else:
                return ConstDictVariable(items, mutation_type=ValueMutationNew())

        if isinstance(arg, dict):
            arg_list = [VariableTracker.build(tx, k) for k in arg]
            return _make_result(dict.fromkeys(arg_list, value))
        elif iterator := generic_getiter(tx, arg):
            keys = unpack_iterable(tx, iterator)
            if all(is_hashable(v) for v in keys):
                return _make_result(dict.fromkeys(keys, value))

        unimplemented(
            gb_type="failed to call dict.fromkeys()",
            context=f"{user_cls.__name__}.fromkeys(): {args} {kwargs}",
            explanation=f"Failed to call {user_cls.__name__}.fromkeys() because "
            "arguments could not be automatically converted to a list, "
            "or some dict key is not hashable.",
            hints=[
                "Manually convert the argument to a list.",
                "Ensure all keys are hashable.",
            ],
        )


class IterBuiltinVariable(BaseBuiltinVariable):
    """Variable tracker for the `iter` builtin."""

    _fn = iter

    def __init__(self, value: Any = iter, **kwargs: Any) -> None:
        if value is not iter:
            raise AssertionError(f"IterBuiltinVariable value must be iter, got {value}")
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return "IterBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Python/bltinmodule.c#L1666-L1682

        if not args:
            raise_observed_exception(
                TypeError,
                tx,
                args=["iter expected at least 1 argument, got 0"],
            )

        if len(args) == 1:
            return generic_getiter(tx, args[0])
        else:
            return variables.UserFunctionVariable(
                polyfills.builtins.callable_iterator
            ).call_function(tx, args, kwargs)


class GetAttrBuiltinVariable(BaseBuiltinVariable):
    """Variable tracker for the `getattr` builtin."""

    _fn = getattr

    def __init__(self, value: Any = getattr, **kwargs: Any) -> None:
        if value is not getattr:
            raise AssertionError(
                f"GetAttrBuiltinVariable value must be getattr, got {value}"
            )
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return "GetAttrBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .lazy import LazyVariableTracker

        if any(isinstance(a, LazyVariableTracker) for a in args):
            args = [
                a.realize() if isinstance(a, LazyVariableTracker) else a for a in args
            ]
        try:
            return self._call_getattr(tx, args, kwargs)
        except Unsupported:
            # Replicate the constant-fold fallback from BuiltinVariable._make_handler:
            # if all args are python constants, evaluate getattr() directly rather
            # than propagating a graph break from var_getattr.
            if not check_unspec_or_constant_args(args, kwargs):
                raise
            try:
                result = getattr(*[a.as_python_constant() for a in args])
            except AttributeError:
                raise_observed_exception(AttributeError, tx)
                raise
            except AsPythonConstantNotImplementedError:
                raise
            except Exception as exc:
                raise_observed_exception(type(exc), tx, args=list(exc.args))
                raise
            return VariableTracker.build(tx, result)

    def _call_getattr(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        obj = args[0]
        name_var = args[1]
        default = args[2] if len(args) > 2 else None

        if not name_var.is_python_constant():
            unimplemented(
                gb_type="getattr() with non-constant name argument",
                context=f"getattr({obj}, {name_var}, {default})",
                explanation="getattr() with non-constant name argument is not supported",
                hints=["Ensure the name argument of getattr() is a string"],
            )

        name = name_var.as_python_constant()

        # See NOTE [Tensor "grad" and "_grad" attr]
        if obj.is_tensor() and name == "_grad":
            name = "grad"

        if tx.output.side_effects.is_attribute_mutation(obj):
            if isinstance(obj, variables.UnspecializedNNModuleVariable):
                if (
                    name
                    in (
                        "named_parameters",
                        "parameters",
                        "named_buffers",
                        "buffers",
                        "named_modules",
                        "modules",
                    )
                    and obj.is_state_mutated
                    and tx.output.side_effects.has_pending_mutation(obj)
                ):
                    unimplemented(
                        gb_type="getattr() on nn.Module with pending mutation",
                        context=f"getattr({obj}, {name}, {default})",
                        explanation="Intentionally graph breaking on getattr() on a nn.Module "
                        "with a pending mutation",
                        hints=[],
                    )

        if tx.output.side_effects.has_pending_mutation_of_attr(obj, name):
            if not isinstance(obj, variables.UserDefinedObjectVariable):
                return tx.output.side_effects.load_attr(obj, name)
            if tx.output.side_effects.has_pending_mutation_of_attr(
                obj, name, AttrMutationKind.INSTANCE_DICT
            ):
                value = tx.output.side_effects.load_attr(obj, name, deleted_ok=True)
                type_attr = obj.lookup_class_mro_attr(name)
                if not isinstance(value, variables.DeletedVariable) and (
                    type_attr is NO_SUCH_SUBOBJ or not is_data_descriptor(type_attr)
                ):
                    return value

        if default is not None:
            hasattr_var = obj.call_obj_hasattr(tx, name)
            if hasattr_var is not None:
                if not hasattr_var.is_constant_match(True, False):
                    raise AssertionError(
                        f"hasattr_var must be a constant True or False, got {hasattr_var}"
                    )
                if not hasattr_var.as_python_constant():
                    return default
            else:
                return default

        source = obj.source and AttrSource(obj.source, name)
        if name in {"__bases__", "__base__", "__flags__"}:
            try:
                value = obj.as_python_constant()
                if isinstance(value, type):
                    if name == "__bases__":
                        tuple_args = [
                            VariableTracker.build(
                                tx, b, source and GetItemSource(source, i)
                            )
                            for i, b in enumerate(value.__bases__)
                        ]
                        return variables.TupleVariable(tuple_args, source=source)
                    if name == "__base__":
                        return VariableTracker.build(tx, value.__base__, source)
                    if name == "__flags__":
                        return VariableTracker.build(tx, value.__flags__)
            except NotImplementedError:
                pass

        if isinstance(obj, variables.NNModuleVariable):
            return obj.var_getattr(tx, name)
        elif isinstance(
            obj,
            (
                variables.TensorVariable,
                variables.NamedTupleVariable,
                variables.ConstantVariable,
                variables.DefaultDictVariable,
                variables.DistributedVariable,
                variables.UserDefinedClassVariable,
                variables.UserDefinedObjectVariable,
            ),
        ):
            if (
                isinstance(obj, variables.UserDefinedObjectVariable)
                and issubclass(obj.value.__class__, unittest.TestCase)
                and config.enable_trace_unittest
                and name
                in (
                    "assertNotWarns",
                    "assertWarnsRegex",
                    "assertWarns",
                )
            ):
                unimplemented(
                    gb_type="Failed to trace unittest method",
                    context=f"function: unittest.TestCase.{name}",
                    explanation=f"Dynamo does not know how to trace unittest method `{name}` ",
                    hints=[
                        f"Avoid calling `TestCase.{name}`. "
                        "Please report an issue to PyTorch.",
                    ],
                )
            if obj.is_tensor():
                # pyrefly: ignore[missing-attribute]
                fake_val = obj.as_proxy().node.meta["example_value"]
                if (
                    isinstance(fake_val, torch.Tensor)
                    and is_sparse_any(fake_val)
                    and (not tx.export or not config.capture_sparse_compute)
                ):
                    unimplemented(
                        gb_type="Attempted to wrap sparse Tensor",
                        context="",
                        explanation="torch.compile does not support sparse Tensors",
                        hints=[*graph_break_hints.SPARSE_TENSOR],
                    )

            try:
                return obj.var_getattr(tx, name)
            except AsPythonConstantNotImplementedError:
                # dont fallback on as_python_constant error because this leads
                # to a failure later on, and leads to a wrong stacktrace
                raise
            except NotImplementedError:
                return variables.GetAttrVariable(obj, name, source=source)
        elif isinstance(obj, variables.TorchInGraphFunctionVariable):
            # Get OpOverload from an OpOverloadPacket, e.g., torch.ops.aten.add.default.
            try:
                member = getattr(obj.value, name)
            except AttributeError:
                raise_observed_exception(AttributeError, tx)
                raise

            if isinstance(
                member, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)
            ) and torch._dynamo.trace_rules.is_aten_op_or_tensor_method(member):
                return variables.TorchInGraphFunctionVariable(member, source=source)
            else:
                return variables.GetAttrVariable(obj, name, source=source)
        elif isinstance(obj, DummyModule):
            # TODO(mlazos) - Do we need this?
            if obj.is_torch or name not in obj.value.__dict__:
                member = getattr(obj.value, name)
            else:
                member = obj.value.__dict__[name]

            if config.replay_record_enabled:
                tx.exec_recorder.record_module_access(obj.value, name, member)  # type: ignore[arg-type, union-attr]
            return VariableTracker.build(tx, member, source)
        else:
            try:
                return obj.var_getattr(tx, name)
            except NotImplementedError:
                return variables.GetAttrVariable(obj, name, source=source)


class HasAttrBuiltinVariable(BaseBuiltinVariable):
    """Variable tracker for the `hasattr` builtin."""

    _fn = hasattr

    def __init__(self, value: Any = hasattr, **kwargs: Any) -> None:
        if value is not hasattr:
            raise AssertionError(
                f"HasAttrBuiltinVariable value must be hasattr, got {value}"
            )
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return "HasAttrBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .lazy import LazyVariableTracker

        if any(isinstance(a, LazyVariableTracker) for a in args):
            args = [
                a.realize() if isinstance(a, LazyVariableTracker) else a for a in args
            ]
        if len(args) != 2 or kwargs:
            raise_observed_exception(TypeError, tx)
        obj, attr = args
        if not attr.is_python_constant():
            raise_observed_exception(TypeError, tx)
        result = obj.call_obj_hasattr(tx, attr.as_python_constant())
        if result is None:
            unimplemented(
                gb_type="hasattr() on unsupported type",
                context=f"hasattr({obj}, {attr})",
                explanation=f"hasattr() is not supported on type {obj.python_type_name()}",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        return result


class SetAttrBuiltinVariable(BaseBuiltinVariable):
    """Variable tracker for the `setattr` builtin."""

    _fn = setattr

    def __init__(self, value: Any = setattr, **kwargs: Any) -> None:
        if value is not setattr:
            raise AssertionError(
                f"SetAttrBuiltinVariable value must be setattr, got {value}"
            )
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return "SetAttrBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .lazy import LazyVariableTracker

        if any(isinstance(a, LazyVariableTracker) for a in args):
            args = [
                a.realize() if isinstance(a, LazyVariableTracker) else a for a in args
            ]
        if len(args) != 3 or kwargs:
            raise_observed_exception(TypeError, tx)
        obj, name_var, val = args
        result = self._call_setattr(tx, obj, name_var, val)
        if result is not None:
            return result
        unimplemented(
            gb_type="setattr() on unsupported type",
            context=f"setattr({obj}, {name_var}, {val})",
            explanation=f"setattr() is not supported on type {obj.python_type_name()}",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def _call_setattr(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        name_var: VariableTracker,
        val: VariableTracker,
    ) -> VariableTracker | None:
        if isinstance(
            obj,
            (
                variables.DefaultDictVariable,
                variables.UserDefinedObjectVariable,
                variables.NestedUserFunctionVariable,
                variables.ExceptionVariable,
                variables.TracebackVariable,
            ),
        ):
            return obj.call_method(tx, "__setattr__", [name_var, val], {})
        elif (
            tx.output.side_effects.is_attribute_mutation(obj)
            and name_var.is_python_constant()
        ):
            name = name_var.as_python_constant()
            if obj.is_tensor():
                from .builder import wrap_fx_proxy

                if name == "requires_grad":
                    # TODO(azahed98): Make it work properly
                    unimplemented(
                        gb_type="setattr() on Tensor.requires_grad",
                        context=f"setattr({obj}, {name}, {val})",
                        explanation="setattr() on Tensor.requires_grad not supported. "
                        "Mutating requires_grad can introduce a new leaf from non-leaf or vice versa in "
                        "the middle of the graph, which AOTAutograd does not currently know how to handle.",
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )
                elif name == "data":
                    # [Note: set_data_on_scoped_tensor]
                    # TODO(azahed98): The plan of record is to introduce a set_data op, entirely subsume the
                    # operation into a call_function in the fx graph, and let aot_autograd handle it.
                    if obj.source is None:
                        unimplemented(
                            gb_type="Failed to mutate tensor data attribute",
                            context=f"setattr({obj}, {name}, {val})",
                            explanation="Dynamo only supports mutating `.data`"
                            " of tensor created outside `torch.compile` region",
                            hints=[
                                "Don't mutate `.data` on this tensor, or move "
                                "the mutation out of `torch.compile` region",
                            ],
                        )
                    elif obj.dtype != val.dtype:  # type: ignore[attr-defined]
                        unimplemented(
                            gb_type="Failed to mutate tensor data attribute to different dtype",
                            context=f"setattr({obj}, {name}, {val})",
                            explanation="Dynamo only supports mutating `.data`"
                            " of tensor to a new one with the same dtype",
                            hints=[
                                "Don't mutate `.data` on this tensor, or move "
                                "the mutation out of `torch.compile` region",
                            ],
                        )

                    # Remove the old reference in tracked fakes - if we don't do this
                    # new .data value size and shape differences will cause
                    # tracked fakes to produce incorrect guards. This is sound because the TensorVariable
                    # coming out of set_() below will be a new one, and get
                    # installed in tracked fakes.
                    to_remove = [
                        tf for tf in tx.output.tracked_fakes if tf.source == obj.source
                    ]
                    for tf in to_remove:
                        tx.output.tracked_fakes.remove(tf)

                    # Step 1 - disable grads
                    with dynamo_disable_grad(tx), torch.no_grad():
                        # Step 2 - call `set_`
                        out = wrap_fx_proxy(
                            tx,
                            tx.output.create_proxy(
                                "call_function",
                                torch.Tensor.set_,
                                *proxy_args_kwargs([obj, val], {}),
                            ),
                        )

                    # Step 3 - drop the version counter - this is a step required to get
                    # .data setting to play correctly with the autograd engine.
                    # Essentially, dynamo is trying to faithfully preserve the (absurd)
                    # behavior of .data= from eager mode
                    def _lower_version_count_by_1(x: torch.Tensor) -> torch.Tensor:
                        version = x._version
                        if version > 0:
                            version = version - 1
                        torch._C._autograd._unsafe_set_version_counter((x,), (version,))
                        return x

                    tx.output.create_proxy(
                        "call_function",
                        _lower_version_count_by_1,
                        (out.as_proxy(),),
                        {},
                    )
                    _lower_version_count_by_1(obj.as_proxy().node.meta["example_value"])
                    # This handles options prop, guards and ends with a clone
                    # Step 4 - replace all reference to the current object with the new one
                    return out
                elif name in ("_grad", "grad"):
                    # NOTE: [Tensor "grad" and "_grad" attr]
                    # _grad and grad share the same setter/getter, see
                    # THPVariable_properties, and here we make sure setting one
                    # enables reading `val` from the other, by routing all
                    # read/write to `grad`.
                    name = "grad"
                elif is_tensor_getset_descriptor(name):
                    # Attribute like `torch.Tensor.real` has special setters we
                    # don't yet support; it's not as simple adding an entry to
                    # the side effect mapping.
                    unimplemented(
                        gb_type="Failed to set tensor attribute",
                        context=f"setattr({obj}, {name}, {val})",
                        explanation="Dynamo doesn't support setting these tensor attributes",
                        hints=[
                            f"Don't mutate attribute '{name}' on tensors, or "
                            "move the mutation out of `torch.compile` region",
                        ],
                    )

            tx.output.side_effects.store_attr(obj, name, val)
            return val
        elif isinstance(obj, variables.NNModuleVariable):
            if not tx.output.is_root_tracer():
                unimplemented(
                    gb_type="nn.Module mutation in HigherOrderOp",
                    context=f"nn.Module: {obj}",
                    explanation="Inplace modifying nn.Module params/buffers inside HigherOrderOps is not allowed.",
                    hints=[
                        "Remove the mutation or move it outside of the HigherOrderOp.",
                        *graph_break_hints.FUNDAMENTAL,
                    ],
                )
            if name_var.is_python_constant() and isinstance(
                val, variables.TensorVariable
            ):
                assigning_fake_val = get_fake_value(val.as_proxy().node, tx)

                try:
                    getattr_var = obj.var_getattr(tx, name_var.as_python_constant())
                except (AttributeError, ObservedAttributeError):
                    getattr_var = None

                if getattr_var is not None and getattr_var.is_tensor():
                    # get_fake_val will get the same fake tensor
                    existing_fake_attr = get_fake_value(getattr_var.as_proxy().node, tx)

                    # same tensor identity, setattr is a no-op
                    mod_setattr = inspect.getattr_static(obj.module_type, "__setattr__")
                    if (
                        existing_fake_attr is assigning_fake_val
                        and mod_setattr is torch.nn.Module.__setattr__
                    ):
                        return getattr_var

            obj.convert_to_unspecialized(tx)
        return None


class ListBuiltinVariable(BaseBuiltinVariable):
    """Variable tracker for the `list` builtin constructor."""

    _fn = list

    def __init__(self, value: type = list, **kwargs: Any) -> None:
        if value is not list:
            raise AssertionError(f"ListBuiltinVariable value must be list, got {value}")
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return "ListBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/listobject.c#L1265-L1287
        if kwargs:
            raise_type_error(
                tx,
                "list() takes no keyword arguments",
            )
        if len(args) == 0:
            return ListVariable([], mutation_type=ValueMutationNew())
        elif len(args) > 1:
            raise_type_error(
                tx,
                f"list expected at most 1 argument, got {len(args)}",
            )

        obj = args[0]
        if obj.source and not is_constant_source(obj.source):
            if isinstance(obj, TupleIteratorVariable):
                install_guard(obj.source.make_guard(GuardBuilder.TUPLE_ITERATOR_LEN))
            elif not isinstance(
                obj,
                (variables.IteratorVariable, variables.LocalGeneratorObjectVariable),
            ):
                if isinstance(
                    obj,
                    (
                        ConstDictVariable,
                        variables.OrderedSetVariable,
                        variables.DictKeySetVariable,
                    ),
                ):
                    tx.output.guard_on_key_order.add(obj.source)
                if isinstance(obj, variables.MappingProxyVariable):
                    install_guard(
                        obj.source.make_guard(GuardBuilder.MAPPING_KEYS_CHECK)
                    )
                elif not isinstance(obj, variables.UnspecializedNNModuleVariable):
                    install_guard(obj.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))

        lst = ListVariable([], mutation_type=ValueMutationNew())
        lst.call_method(tx, "extend", [args[0]], {})
        return lst

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__new__":
            if len(args) == 1 and not kwargs:
                list_vt = ListVariable([], mutation_type=ValueMutationNew())
                if isinstance(args[0], ListBuiltinVariable):
                    return list_vt
                return tx.output.side_effects.track_new_user_defined_object(
                    self,
                    args[0],
                    args[1:],
                    tx=tx,
                )

        return super().call_method(tx, name, args, kwargs)


# pyrefly: ignore [deprecated]
@contextlib.contextmanager
def dynamo_disable_grad(tx: "InstructionTranslatorBase") -> typing.Iterator[None]:
    from . import GradModeVariable

    gmv = GradModeVariable.create(tx, False)
    try:
        gmv.enter(tx)
        yield
    finally:
        gmv.exit(tx)
