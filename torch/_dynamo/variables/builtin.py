# mypy: ignore-errors

import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import types
from collections import defaultdict, OrderedDict
from collections.abc import KeysView
from typing import Dict, List, TYPE_CHECKING

import torch
from torch import sym_float, sym_int
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import config, variables
from ..exc import (
    AttributeMutationError,
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
    TypeSource,
)
from ..utils import (
    check_constant_args,
    check_numpy_ndarray_args,
    check_unspec_or_constant_args,
    check_unspec_python_args,
    does_not_override_dict_iter_methods,
    extract_fake_example_value,
    get_fake_value,
    guard_if_dyn,
    is_wrapper_or_member_descriptor,
    istype,
    numpy_operator_wrapper,
    proxy_args_kwargs,
    tensortype_to_dtype,
)
from .base import ValueMutationNew, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import EventVariable, StreamVariable
from .dicts import (
    ConstDictVariable,
    DefaultDictVariable,
    DictView,
    FrozensetVariable,
    is_hashable,
    SetVariable,
)
from .lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    SizeVariable,
    TupleIteratorVariable,
    TupleVariable,
)
from .tensor import (
    FakeItemVariable,
    supported_comparison_ops,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .user_defined import UserDefinedObjectVariable, UserDefinedVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


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


class BuiltinVariable(VariableTracker):
    _SENTINEL = object()
    _nonvar_fields = {
        "fn",
        *VariableTracker._nonvar_fields,
    }

    @classmethod
    def create_with_source(cls, value, source):
        install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        return cls(value, source=source)

    @staticmethod
    @functools.lru_cache(None)
    def _constant_fold_functions():
        fns = {
            abs,
            all,
            any,
            bool,
            callable,
            chr,
            divmod,
            float,
            getattr,
            int,
            len,
            max,
            min,
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

    def can_constant_fold_through(self):
        return self.fn in self._constant_fold_functions()

    @staticmethod
    @functools.lru_cache(None)
    def _fx_graph_functions():
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
        return fns

    @staticmethod
    @functools.lru_cache(None)
    def _binops():
        # function -> ([forward name, reverse name, in-place name], in-place op)
        fns = {
            operator.add: (["__add__", "__radd__", "__iadd__"], operator.iadd),
            operator.sub: (["__sub__", "__rsub__", "__isub__"], operator.isub),
            operator.mul: (["__mul__", "__rmul__", "__imul__"], operator.imul),
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
            operator.lshift: (
                ["__lshift__", "__rlshift__", "__ilshift__"],
                operator.ilshift,
            ),
            operator.rshift: (
                ["__rshift__", "__rrshift__", "__irshift__"],
                operator.irshift,
            ),
            # NB: The follow binary operators are not supported for now, since the
            # corresponding magic methods aren't defined on SymInt / SymFloat:
            # operator.matmul
            # divmod
            # operator.and_
            # operator.or_
            # operator.xor
        }
        return fns

    @staticmethod
    @functools.lru_cache(None)
    def _binop_handlers():
        # Multiple dispatch mechanism defining custom binop behavior for certain type
        # combinations. Handlers are attempted in order, and will be used if the type checks
        # match. They are expected to have the signature:
        # fn(tx, arg0: VariableTracker, arg1: VariableTracker) -> VariableTracker
        from .dicts import DictKeys, SetVariable
        from .functions import BaseUserFunctionVariable, UserFunctionVariable
        from .nn_module import NNModuleVariable
        from .tensor import supported_const_comparison_ops
        from .torch import BaseTorchVariable
        from .user_defined import (
            UserDefinedClassVariable,
            UserDefinedObjectVariable,
            UserDefinedVariable,
        )

        # Override table contains: op_fn -> [list of handlers]
        op_handlers = {}
        for (
            op,
            (magic_method_names, in_place_op),
        ) in BuiltinVariable._binops().items():
            op_handlers[op] = []
            op_handlers[in_place_op] = []

            forward_name, reverse_name, inplace_name = magic_method_names

            # User-defined args (highest precedence)
            def user_defined_handler(
                tx,
                a,
                b,
                *,
                forward_name=forward_name,
                reverse_name=reverse_name,
            ):
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
                tx: "InstructionTranslator", a, b, *, forward_name=inplace_name
            ):
                return a.call_method(tx, forward_name, [b], {})

            op_handlers[in_place_op].append(
                ((UserDefinedVariable, VariableTracker), user_defined_inplace_handler)
            )
            op_handlers[in_place_op].append(
                ((VariableTracker, UserDefinedVariable), user_defined_inplace_handler)
            )

            # Dynamic shape args
            def dynamic_handler(tx: "InstructionTranslator", a, b, *, fn=op):
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
        def tuple_add_handler(tx: "InstructionTranslator", a, b):
            return TupleVariable([*a.items, *b.unpack_var_sequence(tx)])

        def size_add_handler(tx: "InstructionTranslator", a, b):
            return SizeVariable([*a.items, *b.unpack_var_sequence(tx)])

        list_like_addition_handlers = [
            # NB: Prefer the tuple-specific logic over base logic because of
            # some SizeVariable weirdness. Specifically, the tuple-specific logic
            # drops the subclass type (e.g. SizeVariable) and returns TupleVariables.
            (
                (SizeVariable, SizeVariable),
                size_add_handler,
            ),
            (
                (TupleVariable, TupleVariable),
                tuple_add_handler,
            ),
            (
                (TupleVariable, ConstantVariable),
                tuple_add_handler,
            ),
            (
                (ConstantVariable, TupleVariable),
                lambda tx, a, b: TupleVariable(
                    [*a.unpack_var_sequence(tx), *b.items],
                ),
            ),
            (
                (
                    ListVariable,
                    (BaseListVariable, ConstantVariable, ListIteratorVariable),
                ),
                lambda tx, a, b: ListVariable(
                    [*a.items, *b.unpack_var_sequence(tx)],
                    mutation_type=ValueMutationNew(),
                ),
            ),
            (
                (BaseListVariable, BaseListVariable),
                lambda tx, a, b: type(a)([*a.items, *b.items]),
            ),
        ]
        op_handlers[operator.add].extend(list_like_addition_handlers)

        def list_iadd_handler(tx: "InstructionTranslator", a, b):
            if a.is_immutable() or not b.has_unpack_var_sequence(tx):
                # Handler doesn't apply
                return None

            seq = b.unpack_var_sequence(tx)
            tx.output.side_effects.mutation(a)
            a.items.extend(seq)
            return a

        list_like_iadd_handlers = [
            (
                (ListVariable, VariableTracker),
                list_iadd_handler,
            ),
            (
                (TupleVariable, TupleVariable),
                tuple_add_handler,
            ),
            (
                (TupleVariable, ConstantVariable),
                tuple_add_handler,
            ),
        ]
        op_handlers[operator.iadd].extend(list_like_iadd_handlers)

        # List-like expansion (e.g. [1, 2, 3] * 3)
        def expand_list_like(tx: "InstructionTranslator", lst, const):
            if isinstance(lst, ConstantVariable):
                lst, const = const, lst
            return lst.__class__(
                items=lst.items * const.as_python_constant(),
                mutation_type=ValueMutationNew(),
            )

        list_like_expansion_handlers = [
            ((ListVariable, ConstantVariable), expand_list_like),
            ((TupleVariable, ConstantVariable), expand_list_like),
            ((ConstantVariable, ListVariable), expand_list_like),
            ((ConstantVariable, TupleVariable), expand_list_like),
        ]
        op_handlers[operator.mul].extend(list_like_expansion_handlers)

        size_or_tuple = (SizeVariable, TupleVariable)
        has_set_items = (SetVariable, DictKeys)

        def create_cmp_op_handlers(op):
            def compare_by_value(tx: "InstructionTranslator", a, b):
                return ConstantVariable(op(a.value, b.value))

            result = [((ConstantVariable, ConstantVariable), compare_by_value)]

            if op in supported_const_comparison_ops.values():
                # Tensor is None, List is not None, etc
                none_result = op(object(), None)
                if op.__name__.startswith("is_"):

                    def never(tx: "InstructionTranslator", a, b):
                        return ConstantVariable(none_result)

                    obj_op_none = never
                    none_op_obj = never
                else:

                    def obj_op_none(
                        tx: "InstructionTranslator", a, b: ConstantVariable
                    ):
                        if b.value is None or b.value is True or b.value is False:
                            return ConstantVariable(none_result)

                    def none_op_obj(
                        tx: "InstructionTranslator", a: ConstantVariable, b
                    ):
                        if a.value is None or a.value is True or a.value is False:
                            return ConstantVariable(none_result)

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

            def list_compare_nocheck(tx: "InstructionTranslator", left, right):
                return BaseListVariable.list_compare(tx, op, left, right)

            def list_compare_check(tx: "InstructionTranslator", left, right):
                if type(left) is not type(
                    right
                ):  # Mismatch in BaseListVariable subclasses
                    unimplemented(f"{op.__name__}({left}, {right})")
                return BaseListVariable.list_compare(tx, op, left, right)

            def compare_set_items(tx: "InstructionTranslator", left, right):
                return ConstantVariable(op(left.set_items, right.set_items))

            def compare_via_method(tx: "InstructionTranslator", left, right):
                return left.call_method(tx, f"__{op.__name__}__", [right], {})

            if op.__name__.startswith("is_"):
                compare_user_defined = compare_by_value
            else:
                compare_user_defined = compare_via_method

            op_var = BuiltinVariable(op)
            result.extend(
                [
                    (
                        (
                            (UserFunctionVariable, BuiltinVariable),
                            (UserFunctionVariable, BuiltinVariable),
                        ),
                        lambda tx, a, b: ConstantVariable(op(a.fn, b.fn)),
                    ),
                    (
                        (
                            NNModuleVariable,
                            NNModuleVariable,
                        ),
                        lambda tx, a, b: ConstantVariable(
                            op(
                                tx.output.get_submodule(a.module_key),
                                tx.output.get_submodule(b.module_key),
                            )
                        ),
                    ),
                    ((size_or_tuple, size_or_tuple), list_compare_nocheck),
                    (
                        (variables.BaseListVariable, variables.BaseListVariable),
                        list_compare_check,
                    ),
                    ((has_set_items, has_set_items), compare_set_items),
                    (
                        (UserDefinedObjectVariable, UserDefinedObjectVariable),
                        compare_user_defined,
                    ),
                    (
                        (UserDefinedClassVariable, UserDefinedClassVariable),
                        compare_user_defined,
                    ),
                    (
                        (
                            (StreamVariable, EventVariable, ConstantVariable),
                            (StreamVariable, EventVariable, ConstantVariable),
                        ),
                        compare_by_value,
                    ),
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

            if op.__name__.startswith("is_"):

                def handle_is(tx: "InstructionTranslator", left, right):
                    # If the two objects are of different type, we can safely return False
                    # and True for `is` and `is not`, respectively
                    if type(left) is not type(right):
                        return ConstantVariable.create(op.__name__ != "is_")

                result.append(((VariableTracker, VariableTracker), handle_is))

            return result

        for op in supported_comparison_ops.values():
            assert callable(op)
            assert op not in op_handlers
            op_handlers[op] = create_cmp_op_handlers(op)

        return op_handlers

    @staticmethod
    def _find_binop_handler(op, a_type, b_type):
        handlers = BuiltinVariable._binop_handlers().get(op)
        if handlers is None:
            return None

        matches = []
        for (type1, type2), handler in handlers:
            if issubclass(a_type, type1) and issubclass(b_type, type2):
                matches.append(handler)
        return matches

    def can_insert_in_graph(self):
        return self.fn in self._fx_graph_functions()

    def __init__(self, fn, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fn = fn

    def __repr__(self) -> str:
        if self.fn is None:
            name = "None"
        else:
            name = self.fn.__name__

        return f"{self.__class__.__name__}({name})"

    def as_python_constant(self):
        return self.fn

    def as_proxy(self):
        DTYPE = {
            bool: torch.bool,
            int: torch.int64,
            float: torch.float64,
        }
        if self.fn in DTYPE:
            return DTYPE[self.fn]
        return super().as_proxy()

    def reconstruct(self, codegen):
        name = self.fn.__name__
        assert self.fn.__module__ == "builtins"
        assert name not in codegen.tx.f_globals, "shadowed global"
        codegen.append_output(codegen.create_load_global(name, False, add=True))

    def constant_args(self, *args, **kwargs):
        return check_constant_args(args, kwargs)

    def tensor_args(self, *args):
        any_tensor = False
        for arg in args:
            if isinstance(arg, variables.GetAttrVariable):
                return False
            any_tensor = any_tensor or isinstance(arg, variables.TensorVariable)
        return any_tensor

    def tensor_args_type(self, arg_types):
        any_tensor = False
        for arg_type in arg_types:
            if issubclass(arg_type, variables.GetAttrVariable):
                return False
            any_tensor = any_tensor or issubclass(arg_type, variables.TensorVariable)
        return any_tensor

    def python_and_tensor_constant_only(self, *args, **kwargs):
        tensor_args = []
        non_tensor_args = []
        for i in itertools.chain(args, kwargs.values()):
            if isinstance(i, variables.TensorVariable):
                tensor_args.append(i)
            else:
                non_tensor_args.append(i)
        return all(
            is_constant_source(t.source) if t.source is not None else False
            for t in tensor_args
        ) and self.constant_args(*non_tensor_args)

    @staticmethod
    def unwrap_unspec_args_kwargs(args, kwargs):
        return [x.as_python_constant() for x in args], {
            k: v.as_python_constant() for k, v in kwargs.items()
        }

    def has_constant_handler(self, args, kwargs):
        return self.can_constant_fold_through() and check_unspec_or_constant_args(
            args, kwargs
        )

    @staticmethod
    def _make_handler(fn, arg_types: List[type], has_kwargs: bool):
        from .lazy import LazyVariableTracker

        obj = BuiltinVariable(fn)
        handlers = []

        if any(issubclass(t, LazyVariableTracker) for t in arg_types):
            return lambda tx, args, kwargs: obj.call_function(
                tx, [v.realize() for v in args], kwargs
            )

        if inspect.isclass(fn) and issubclass(fn, Exception):

            def create_exception_class_object(
                tx: "InstructionTranslator", args, kwargs
            ):
                if fn is AssertionError and not all(
                    isinstance(x, variables.ConstantVariable)
                    and isinstance(x.value, str)
                    for x in args
                ):
                    unimplemented("assert with non-string message")

                return variables.ExceptionVariable(fn, args, **kwargs)

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

                def call_binop_handlers(tx: "InstructionTranslator", args, _):
                    for fn in binop_handlers:
                        rv = fn(tx, *args)
                        if rv:
                            return rv

                handlers.append(call_binop_handlers)

        self_handler = getattr(obj, f"call_{fn.__name__}", None)
        if self_handler:

            def call_self_handler(tx: "InstructionTranslator", args, kwargs):
                try:
                    result = self_handler(tx, *args, **kwargs)
                    if result is not None:
                        return result
                except TypeError:
                    # Check if binding is bad. inspect signature bind is expensive.
                    # So check only when handler call fails.
                    try:
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
                                f"invalid handler args {self_handler} {args} {kwargs}"
                            )
                    else:
                        raise
                except Unsupported as exc:
                    has_constant_handler = obj.has_constant_handler(args, kwargs)
                    if not has_constant_handler:
                        raise
                    # Actually, we will handle this just fine
                    exc.remove_from_stats()

            handlers.append(call_self_handler)

        if obj.can_constant_fold_through():
            if (
                all(issubclass(x, ConstantVariable) for x in arg_types)
                and not has_kwargs
            ):

                def constant_fold_handler(tx: "InstructionTranslator", args, kwargs):
                    # fast path
                    try:
                        res = fn(
                            *[x.as_python_constant() for x in args],
                        )
                    except Exception as exc:
                        unimplemented(f"constant fold exception: {repr(exc)}")
                    return VariableTracker.build(tx, res)

            else:

                def constant_fold_handler(tx: "InstructionTranslator", args, kwargs):
                    # path with a runtime check
                    if check_unspec_or_constant_args(args, kwargs):
                        try:
                            res = fn(
                                *[x.as_python_constant() for x in args],
                                **{
                                    k: v.as_python_constant() for k, v in kwargs.items()
                                },
                            )
                        except Exception as exc:
                            unimplemented(f"constant fold exception: {repr(exc)}")
                        return VariableTracker.build(tx, res)

            handlers.append(constant_fold_handler)

        error_msg = f"builtin: {fn.__name__} {arg_types} {has_kwargs}"
        if len(handlers) == 0:
            return lambda *args: unimplemented(error_msg)
        elif len(handlers) == 1:
            (handler,) = handlers

            def builtin_dispatch(tx: "InstructionTranslator", args, kwargs):
                rv = handler(tx, args, kwargs)
                if rv:
                    return rv
                unimplemented(error_msg)

        else:

            def builtin_dispatch(tx: "InstructionTranslator", args, kwargs):
                for fn in handlers:
                    rv = fn(tx, args, kwargs)
                    if rv:
                        return rv
                unimplemented(error_msg)

        return builtin_dispatch

    def _handle_insert_op_in_graph(self, tx: "InstructionTranslator", args, kwargs):
        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls

        if kwargs and not self.tensor_args(*args, *kwargs.values()):
            return

        # insert handling for torch function here
        from .builder import SourcelessBuilder
        from .torch_function import (
            BUILTIN_TO_TENSOR_FN_MAP,
            BUILTIN_TO_TENSOR_RFN_MAP,
            can_dispatch_torch_function,
            dispatch_torch_function,
        )

        if can_dispatch_torch_function(tx, args, kwargs):
            # Only remap the fn to tensor methods if we aren't exporting
            # export serde does not handle method descriptors today
            if not tx.export:
                # Use sourceless builder, we built the map ourselves
                if not isinstance(args[0], TensorVariable):
                    if self.fn in BUILTIN_TO_TENSOR_RFN_MAP:
                        func = BUILTIN_TO_TENSOR_RFN_MAP[self.fn]
                    else:
                        func = BUILTIN_TO_TENSOR_FN_MAP[self.fn]

                    tmp = args[0]
                    # swap args and call reverse version of func
                    args[0] = args[1]
                    args[1] = tmp
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
                fn, args = IN_PLACE_DESUGARING_MAP[fn], [args[0], args[1]]

            if fn is operator.getitem and isinstance(args[1], SymNodeVariable):
                # Standard indexing will force specialization due to
                # __index__.  Rewrite as a regular torch op which will
                # trace fine
                fn, args = torch.select, [
                    args[0],
                    variables.ConstantVariable.create(0),
                    args[1],
                ]

            # Interaction between ndarray and tensors:
            #   We prefer the tensor op whenever there are tensors involved
            if check_numpy_ndarray_args(args, kwargs) and not any(
                type(arg) == variables.TensorVariable for arg in args
            ):
                proxy = tx.output.create_proxy(
                    "call_function",
                    numpy_operator_wrapper(fn),
                    *proxy_args_kwargs(args, kwargs),
                )

                return wrap_fx_proxy_cls(variables.NumpyNdarrayVariable, tx, proxy)

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
                    args[0] = args[0].convert_to_constant(tx)
                return wrap_fx_proxy(tx, proxy)

        except NotImplementedError:
            unimplemented(f"partial tensor op: {self} {args} {kwargs}")

    call_function_handler_cache = {}

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if kwargs:
            kwargs = {k: v.realize() for k, v in kwargs.items()}
            key = (self.fn, *(type(x) for x in args), True)
        else:
            key = (self.fn, *(type(x) for x in args))

        handler = self.call_function_handler_cache.get(key)
        if not handler:
            self.call_function_handler_cache[key] = handler = self._make_handler(
                self.fn, [type(x) for x in args], bool(kwargs)
            )
        return handler(tx, args, kwargs)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if self.fn is object and name == "__setattr__":
            assert len(args) == 3
            assert len(kwargs) == 0
            obj, name_var, val = args
            obj = obj.realize()
            if (
                isinstance(obj, UserDefinedObjectVariable)
                and tx.output.side_effects.is_attribute_mutation(obj)
                and name_var.is_python_constant()
            ):
                return obj.method_setattr_standard(tx, name_var, val)
        if self.fn is object and name == "__new__":
            assert len(args) == 1
            assert len(kwargs) == 0
            return tx.output.side_effects.track_object_new_from_user_defined_class(
                args[0]
            )
        if self.fn is dict and name == "fromkeys":
            return BuiltinVariable.call_custom_dict_fromkeys(tx, dict, *args, **kwargs)
        return super().call_method(tx, name, args, kwargs)

    def _call_int_float(self, tx: "InstructionTranslator", arg):
        # Handle cases like int(torch.seed())
        # Also handle sym_float to sym_int cases
        if isinstance(arg, (SymNodeVariable, variables.TensorVariable)):
            if isinstance(arg, variables.TensorVariable):
                item = arg.call_method(tx, "item", [], {})
            else:
                item = arg
            fn_ = sym_int if self.fn is int else sym_float
            from torch._dynamo.variables.builder import wrap_fx_proxy

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    fn_,
                    (item.as_proxy(),),
                    {},
                ),
            )

    call_int = _call_int_float
    call_float = _call_int_float

    def call_str(self, tx: "InstructionTranslator", arg):
        # Handle `str` on a user defined function or object
        if isinstance(arg, (variables.UserFunctionVariable)):
            return variables.ConstantVariable.create(value=str(arg.fn))
        elif isinstance(arg, (variables.UserDefinedObjectVariable)):
            # Check if object has __str__ method
            if hasattr(arg.value, "__str__"):
                str_method = arg.value.__str__
            elif hasattr(arg.value, "__repr__"):
                # account for __repr__ functions when __str__ is absent
                str_method = arg.value.__repr__
            else:
                unimplemented("user defined object has no __str__ or __repr__ method")

            if type(arg.value).__str__ is object.__str__:
                # Rely on the object str method
                try:
                    return variables.ConstantVariable.create(value=str_method())
                except AttributeError:
                    # Graph break
                    return
            elif is_wrapper_or_member_descriptor(str_method):
                unimplemented(f"{type(arg.value)} has a C/C++ based str method")
            else:
                # Overrides for custom str method
                # Pass method as function to call tx.inline_user_function_return
                bound_method = str_method.__func__

                try:
                    # Only supports certain function types
                    user_func_variable = variables.UserFunctionVariable(bound_method)
                except AssertionError as e:
                    # Won't be able to do inline the str method, return to avoid graph break
                    log.warning("Failed to create UserFunctionVariable: %s", e)
                    return

                # Inline the user function
                return tx.inline_user_function_return(user_func_variable, [arg], {})

    def _call_min_max(self, tx: "InstructionTranslator", *args):
        if len(args) == 1 and args[0].has_force_unpack_var_sequence(tx):
            items = args[0].force_unpack_var_sequence(tx)
            return self._call_min_max_seq(tx, items)
        elif len(args) == 2:
            return self._call_min_max_binary(tx, args[0], args[1])
        elif len(args) > 2:
            return self._call_min_max_seq(tx, args)

    def _call_min_max_seq(self, tx: "InstructionTranslator", items):
        assert len(items) > 0
        if len(items) == 1:
            return items[0]

        return functools.reduce(functools.partial(self._call_min_max_binary, tx), items)

    def _call_min_max_binary(self, tx: "InstructionTranslator", a, b):
        if a is None or b is None:
            # a or b could be none if we reduce and _call_min_max_binary failed
            # to return something
            return
        if self.tensor_args(a, b):
            if not isinstance(a, variables.TensorVariable):
                a, b = b, a
            assert isinstance(a, variables.TensorVariable)

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

                    fn = {max: np.maximum, min: np.minimum}[self.fn]
                    fn = variables.NumpyVariable(fn)
                else:
                    fn = {max: torch.maximum, min: torch.minimum}[self.fn]
                    fn = variables.TorchInGraphFunctionVariable(fn)
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
                    return variables.FakeItemVariable.from_tensor_variable(result)

                if b.is_python_constant():
                    raw_b = b.as_python_constant()
                else:
                    raw_b = b.raw_value
                if self.fn is max:
                    raw_res = max(a.raw_value, raw_b)
                else:
                    raw_res = min(a.raw_value, raw_b)

                need_unwrap = any(
                    x.need_unwrap
                    for x in [a, b]
                    if isinstance(x, variables.UnspecializedPythonVariable)
                )
                return variables.UnspecializedPythonVariable.from_tensor_variable(
                    result, raw_res, need_unwrap
                )
            # otherwise return tensor
            else:
                return result
        elif isinstance(a, SymNodeVariable) or isinstance(b, SymNodeVariable):
            fn = torch.sym_max if self.fn is max else torch.sym_min
            proxy = tx.output.create_proxy(
                "call_function", fn, *proxy_args_kwargs([a, b], {})
            )
            return SymNodeVariable.create(tx, proxy, None)

    call_min = _call_min_max
    call_max = _call_min_max

    def call_abs(self, tx: "InstructionTranslator", arg: "VariableTracker"):
        # Call arg.__abs__()
        abs_method = BuiltinVariable(getattr).call_function(
            tx, [arg, ConstantVariable.create("__abs__")], {}
        )
        return abs_method.call_function(tx, [], {})

    def call_pos(self, tx: "InstructionTranslator", arg: "VariableTracker"):
        # Call arg.__pos__()
        pos_method = BuiltinVariable(getattr).call_function(
            tx, [arg, ConstantVariable.create("__pos__")], {}
        )
        return pos_method.call_function(tx, [], {})

    def call_index(self, tx: "InstructionTranslator", arg: "VariableTracker"):
        if isinstance(arg, variables.TensorVariable):
            unimplemented("unsupported index(tensor)")

        arg = guard_if_dyn(arg)
        constant_value = operator.index(arg)
        return variables.ConstantVariable.create(constant_value)

    def call_round(self, tx: "InstructionTranslator", arg, *args, **kwargs):
        # Call arg.__round__()
        round_method = BuiltinVariable(getattr).call_function(
            tx, [arg, ConstantVariable.create("__round__")], {}
        )
        return round_method.call_function(tx, args, kwargs)

    def call_range(self, tx: "InstructionTranslator", *args):
        if check_unspec_or_constant_args(args, {}):
            return variables.RangeVariable(args)
        elif self._dynamic_args(*args):
            args = [
                variables.ConstantVariable.create(guard_if_dyn(arg)) for arg in args
            ]
            return variables.RangeVariable(args)
        # None no-ops this handler and lets the driving function proceed
        return None

    def _dynamic_args(self, *args, **kwargs):
        return any(isinstance(x, SymNodeVariable) for x in args) or any(
            isinstance(x, SymNodeVariable) for x in kwargs.values()
        )

    def call_slice(self, tx: "InstructionTranslator", *args):
        return variables.SliceVariable(args)

    def _dyn_proxy(self, tx: "InstructionTranslator", *args, **kwargs):
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_function", self.fn, *proxy_args_kwargs(args, kwargs)
            ),
        )

    # NOTE must handle IteratorVariable separately!
    def _call_iter_tuple_list(
        self, tx: "InstructionTranslator", obj=None, *args, **kwargs
    ):
        assert not isinstance(obj, variables.IteratorVariable)

        if self._dynamic_args(*args, **kwargs):
            return self._dyn_proxy(tx, *args, **kwargs)

        cls = variables.BaseListVariable.cls_for(self.fn)
        if obj is None:
            return cls(
                [],
                mutation_type=ValueMutationNew(),
            )
        elif obj.has_unpack_var_sequence(tx):
            if obj.source and not is_constant_source(obj.source):
                if isinstance(obj, TupleIteratorVariable):
                    install_guard(
                        obj.source.make_guard(GuardBuilder.TUPLE_ITERATOR_LEN)
                    )
                else:
                    if (
                        getattr(obj, "source", False)
                        and isinstance(obj, ConstDictVariable)
                        and not istype(obj, SetVariable)
                    ):
                        tx.output.guard_on_key_order.add(obj.source.name())

                    install_guard(obj.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))

            return cls(
                list(obj.unpack_var_sequence(tx)),
                mutation_type=ValueMutationNew(),
            )

    def _call_tuple_list(self, tx, obj=None, *args, **kwargs):
        if isinstance(obj, variables.IteratorVariable):
            cls = variables.BaseListVariable.cls_for(self.fn)
            return cls(
                list(obj.force_unpack_var_sequence(tx)),
                mutation_type=ValueMutationNew(),
            )
        else:
            return self._call_iter_tuple_list(tx, obj, *args, **kwargs)

    def call_iter(self, tx: "InstructionTranslator", obj, *args, **kwargs):
        if isinstance(obj, variables.IteratorVariable):
            ret = obj
        else:
            # Handle the case where we are iterating over a tuple, list or iterator
            ret = self._call_iter_tuple_list(tx, obj, *args, **kwargs)

        if ret is None:
            # If the object doesn't implement a __iter__ method, it will be an error in eager mode when calling iter on it anyway.
            # If the object implements a __iter__ method, inlining effectively forwards the call to another iter call
            # (e.g. when __iter__ just returns iter(self.list)) or return a user-defined iterator.
            return obj.call_method(tx, "__iter__", args, kwargs)
        return ret

    call_tuple = _call_tuple_list
    call_list = _call_tuple_list

    def call_callable(self, tx: "InstructionTranslator", arg):
        from .functions import BaseUserFunctionVariable
        from .nn_module import NNModuleVariable

        if isinstance(
            arg,
            (
                variables.UserDefinedClassVariable,
                BaseUserFunctionVariable,
                NNModuleVariable,
            ),
        ):
            return variables.ConstantVariable.create(True)
        elif isinstance(arg, UserDefinedVariable):
            return variables.ConstantVariable.create(callable(arg.value))
        elif isinstance(
            arg,
            (
                ConstantVariable,
                SymNodeVariable,
                TensorVariable,
                ListVariable,
                TupleVariable,
                ListIteratorVariable,
            ),
        ):
            return variables.ConstantVariable.create(False)

    def call_cast(self, _, *args, **kwargs):
        if len(args) == 2:
            return args[1]

        unimplemented(f"unsupported args to builtin cast(): {args} {kwargs}")

    def call_dict(self, tx: "InstructionTranslator", *args, **kwargs):
        return BuiltinVariable.call_custom_dict(tx, dict, *args, **kwargs)

    @staticmethod
    def call_custom_dict(tx: "InstructionTranslator", user_cls, *args, **kwargs):
        if not kwargs:
            if not args:
                args = ({},)
            assert len(args) == 1
            arg = args[0]
            if isinstance(arg, dict):
                return ConstDictVariable(
                    arg, user_cls, mutation_type=ValueMutationNew()
                )
            elif isinstance(arg, variables.ConstDictVariable):
                return arg.clone(
                    user_cls=user_cls, source=None, mutation_type=ValueMutationNew()
                )
            elif isinstance(
                arg,
                (
                    ListVariable,
                    TupleVariable,
                    ListIteratorVariable,
                    variables.IteratorVariable,
                ),
            ):
                items = dict(
                    x.force_unpack_var_sequence(tx)
                    for x in arg.force_unpack_var_sequence(tx)
                )
                return ConstDictVariable(
                    items, user_cls, mutation_type=ValueMutationNew()
                )
            elif isinstance(arg, variables.MutableMappingVariable):
                # This is applicable for user defined objects which seem like dict, but are not really dicts. For
                # example, TensorDict derives from MutableMapping. For such cases, we can directly inline the .items
                # method and create a new dict.
                if does_not_override_dict_iter_methods(type(arg.value)):
                    # These are implemeted in C, so we will have to manually construct the items

                    if tx.output.side_effects.has_pending_mutation(arg):
                        unimplemented(
                            f"{user_cls.__name__}.items(): {args} {kwargs} - object is mutated"
                        )

                    new_dict = dict(arg.value.items())
                    return VariableTracker.build(tx, new_dict)
                else:
                    func_var = arg.var_getattr(tx, "items")
                    if not isinstance(func_var, variables.UserFunctionVariable):
                        unimplemented(f"{user_cls.__name__}.items(): {args} {kwargs}")
                    out = tx.inline_user_function_return(func_var, args, kwargs)
                    if isinstance(out, ConstDictVariable):
                        return out
                    return BuiltinVariable(user_cls).call_custom_dict(tx, user_cls, out)
        elif not args and kwargs:
            items = {ConstantVariable.create(k): v for k, v in kwargs.items()}
            return variables.ConstDictVariable(
                items, user_cls=user_cls, mutation_type=ValueMutationNew()
            )
        unimplemented(f"{user_cls.__name__}(): {args} {kwargs}")

    @staticmethod
    def call_custom_dict_fromkeys(
        tx: "InstructionTranslator", user_cls, *args, **kwargs
    ):
        assert user_cls in {dict, OrderedDict, defaultdict}
        if kwargs:
            # Only `OrderedDict.fromkeys` accepts `value` passed by keyword
            assert user_cls is OrderedDict
            assert len(args) == 1 and len(kwargs) == 1 and "value" in kwargs
            args = (*args, kwargs.pop("value"))
        if len(args) == 0:
            raise UserError(TypeError, "fromkeys expected at least 1 argument, got 0")
        if len(args) == 1:
            args = (*args, ConstantVariable.create(None))
        assert len(args) == 2
        arg, value = args
        DictVariableType = (
            ConstDictVariable if user_cls is not defaultdict else DefaultDictVariable
        )

        if isinstance(arg, dict):
            arg = [ConstantVariable.create(k) for k in arg.keys()]
            return DictVariableType(
                dict.fromkeys(arg, value), user_cls, mutation_type=ValueMutationNew()
            )
        elif arg.has_force_unpack_var_sequence(tx):
            keys = arg.force_unpack_var_sequence(tx)
            if all(is_hashable(v) for v in keys):
                return DictVariableType(
                    dict.fromkeys(keys, value),
                    user_cls,
                    mutation_type=ValueMutationNew(),
                )
        unimplemented(f"{user_cls.__name__}.fromkeys(): {args} {kwargs}")

    def call_set(self, tx: "InstructionTranslator", *args, **kwargs):
        # Can we merge this implementation and call_dict's one?
        assert not kwargs
        if not args:
            return SetVariable([], mutation_type=ValueMutationNew())
        assert len(args) == 1
        arg = args[0]
        if isinstance(arg, variables.SetVariable):
            return arg.clone(mutation_type=ValueMutationNew())
        elif arg.has_force_unpack_var_sequence(tx):
            items = arg.force_unpack_var_sequence(tx)
            return SetVariable(items, mutation_type=ValueMutationNew())
        elif isinstance(arg, variables.UserDefinedObjectVariable) and isinstance(
            arg.value, KeysView
        ):
            iter_fn = arg.var_getattr(tx, "__iter__")
            if isinstance(iter_fn, variables.UserMethodVariable):
                out = tx.inline_user_function_return(iter_fn, args, kwargs)
                if isinstance(out, SetVariable):
                    return out
                return BuiltinVariable(set).call_set(tx, out)
            else:
                unimplemented(f"set(): {args} {kwargs}")
        else:
            unimplemented(f"set(): {args} {kwargs}")

    def call_frozenset(self, tx: "InstructionTranslator", *args, **kwargs):
        assert not kwargs
        if not args:
            return FrozensetVariable([])
        assert len(args) == 1
        arg = args[0]
        if isinstance(arg, variables.FrozensetVariable):
            return FrozensetVariable([x.vt for x in arg.set_items])
        elif arg.has_unpack_var_sequence(tx):
            items = arg.unpack_var_sequence(tx)
            return FrozensetVariable(items)
        else:
            unimplemented(f"frozenset(): {args} {kwargs}")

    def call_zip(self, tx: "InstructionTranslator", *args, **kwargs):
        if kwargs:
            assert len(kwargs) == 1 and "strict" in kwargs
        strict = kwargs.pop("strict", False)
        args = [
            arg.unpack_var_sequence(tx) if arg.has_unpack_var_sequence(tx) else arg
            for arg in args
        ]
        return variables.ZipVariable(
            args, strict=strict, mutation_type=ValueMutationNew()
        )

    def call_len(self, tx: "InstructionTranslator", *args, **kwargs):
        return args[0].call_method(tx, "__len__", args[1:], kwargs)

    def call_getitem(self, tx: "InstructionTranslator", *args, **kwargs):
        return args[0].call_method(tx, "__getitem__", args[1:], kwargs)

    def call_isinstance(self, tx: "InstructionTranslator", arg, isinstance_type):
        try:
            arg_type = arg.python_type()
        except NotImplementedError:
            unimplemented(
                f"isinstance({arg}, {isinstance_type}): can't determine type of {arg}"
            )

        isinstance_type = isinstance_type.as_python_constant()

        if isinstance(arg, variables.TensorVariable) and arg.dtype is not None:

            def _tensor_isinstance(tensor_var, tensor_type):
                def check_type(ty):
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
                    return arg.dtype in dtypes

                if type(tensor_type) is tuple:
                    return any(check_type(ty) for ty in tensor_type)
                else:
                    return check_type(tensor_type)

            return variables.ConstantVariable.create(
                _tensor_isinstance(arg, isinstance_type)
            )
        # UserDefinedObject with C extensions can have torch.Tensor attributes,
        # so break graph.
        if isinstance(arg, variables.UserDefinedObjectVariable) and isinstance(
            arg.value, types.MemberDescriptorType
        ):
            unimplemented(
                f"isinstance called on UserDefinedClass {arg} {isinstance_type}"
            )
        # handle __instancecheck__ defined in user class
        if (
            isinstance(arg, variables.UserDefinedObjectVariable)
            and "__instancecheck__" in isinstance_type.__class__.__dict__
        ):
            return variables.ConstantVariable.create(
                isinstance_type.__class__.__instancecheck__(isinstance_type, arg.value)
            )

        try:
            val = issubclass(arg_type, isinstance_type)
        except TypeError:
            val = arg_type is isinstance_type
        return variables.ConstantVariable.create(val)

    def call_issubclass(self, tx: "InstructionTranslator", left_ty, right_ty):
        """Checks if first arg is subclass of right arg"""
        try:
            left_ty_py = left_ty.as_python_constant()
            right_ty_py = right_ty.as_python_constant()
        except NotImplementedError:
            unimplemented(
                f"call_issubclass args not constant left_ty: {left_ty}, right_ty: {right_ty}"
            )

        return variables.ConstantVariable(issubclass(left_ty_py, right_ty_py))

    def call_super(self, tx: "InstructionTranslator", a, b):
        return variables.SuperVariable(a, b)

    def call_next(self, tx: "InstructionTranslator", arg: VariableTracker):
        try:
            return arg.next_variable(tx)
        except Unsupported as ex:
            if isinstance(arg, variables.BaseListVariable):
                ex.remove_from_stats()
                return arg.items[0]
            raise

    def call_hasattr(self, tx: "InstructionTranslator", obj, attr):
        if attr.is_python_constant():
            name = attr.as_python_constant()
            if isinstance(obj, variables.BuiltinVariable):
                return variables.ConstantVariable(hasattr(obj.fn, name))
            return obj.call_hasattr(tx, name)

    def call_map(self, tx: "InstructionTranslator", fn, *seqs):
        seqs = [
            seq.unpack_var_sequence(tx) if seq.has_unpack_var_sequence(tx) else seq
            for seq in seqs
        ]
        return variables.MapVariable(fn, seqs, mutation_type=ValueMutationNew())

    def call_filter(self, tx: "InstructionTranslator", fn, seq):
        seq = seq.unpack_var_sequence(tx) if seq.has_unpack_var_sequence(tx) else seq
        return variables.FilterVariable(fn, seq, mutation_type=ValueMutationNew())

    def call_getattr(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker,
        name_var: VariableTracker,
        default=None,
    ):
        from .. import trace_rules
        from . import (
            ConstantVariable,
            GetAttrVariable,
            TorchInGraphFunctionVariable,
            UserFunctionVariable,
        )

        name = name_var.as_python_constant()

        if not name_var.is_python_constant():
            unimplemented("non-const getattr() name")

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
                        f"pending mutation on nn module, so graph breaking at {name!r} call"
                    )

        if tx.output.side_effects.has_pending_mutation_of_attr(obj, name):
            return tx.output.side_effects.load_attr(obj, name)

        if default is not None:
            hasattr_var = self.call_hasattr(tx, obj, name_var)
            assert hasattr_var.as_python_constant() in (True, False)
            if not hasattr_var.as_python_constant():
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
                        return ConstantVariable.create(value.__flags__)
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
                variables.DistributedVariable,
                variables.UserDefinedClassVariable,
                variables.UserDefinedObjectVariable,
            ),
        ):
            try:
                return obj.var_getattr(tx, name)
            except NotImplementedError:
                return GetAttrVariable(obj, name, source=source)
        elif isinstance(obj, TorchInGraphFunctionVariable):
            # Get OpOverload from an OpOverloadPacket, e.g., torch.ops.aten.add.default.
            member = getattr(obj.value, name)
            if isinstance(
                member, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)
            ) and trace_rules.is_aten_op_or_tensor_method(member):
                return TorchInGraphFunctionVariable(member, source=source)
        elif isinstance(obj, DummyModule):
            # TODO(mlazos) - Do we need this?
            if obj.is_torch or name not in obj.value.__dict__:
                member = getattr(obj.value, name)
            else:
                member = obj.value.__dict__[name]

            if config.replay_record_enabled:
                tx.exec_recorder.record_module_access(obj.value, name, member)
            return VariableTracker.build(tx, member, source)

        elif istype(obj, UserFunctionVariable) and name in ("__name__", "__module__"):
            return ConstantVariable.create(getattr(obj.fn, name))
        else:
            try:
                return obj.var_getattr(tx, name)
            except NotImplementedError:
                return GetAttrVariable(obj, name, source=source)

    def call_setattr(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker,
        name_var: VariableTracker,
        val: VariableTracker,
    ):
        if isinstance(
            obj,
            (
                variables.CustomizedDictVariable,
                variables.PlacementVariable,
                variables.NamedTupleVariable,
                variables.UserDefinedObjectVariable,
            ),
        ):
            return obj.call_method(tx, "__setattr__", [name_var, val], {})
        elif (
            tx.output.side_effects.is_attribute_mutation(obj)
            and name_var.is_python_constant()
        ):
            name = name_var.as_python_constant()
            if isinstance(obj, variables.TensorVariable):
                from .builder import wrap_fx_proxy

                if name == "requires_grad":
                    # TODO(voz): Make it work properly
                    unimplemented(
                        "mutating requires_grad can introduce a new leaf from non-leaf or vice versa in "
                        "the middle of the graph, which aot_autograd does not currently know how to handle. "
                    )
                if name == "data":
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
                    def _lower_version_count_by_1(x):
                        version = x._version
                        if version > 0:
                            version = version - 1
                        torch._C._autograd._unsafe_set_version_counter(x, version)
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

            tx.output.side_effects.store_attr(obj, name, val)
            if name == "_grad":
                tx.output.side_effects.store_attr(obj, "grad", val)

            return val
        elif isinstance(obj, variables.UserDefinedObjectVariable):
            unimplemented(
                f"setattr(UserDefinedObjectVariable) {type(obj.value).__setattr__}"
            )
        elif isinstance(obj, variables.NNModuleVariable):
            if not tx.output.is_root_tracer():
                raise AttributeMutationError(
                    "Can't inplace modify module params/buffers inside HigherOrderOp"
                )
            if name_var.is_python_constant() and isinstance(
                val, variables.TensorVariable
            ):
                assigning_fake_val = get_fake_value(val.as_proxy().node, tx)

                try:
                    getattr_var = obj.var_getattr(tx, name_var.as_python_constant())
                except AttributeError:
                    getattr_var = None

                if isinstance(getattr_var, variables.TensorVariable):
                    # get_fake_val will get the same fake tensor
                    existing_fake_attr = get_fake_value(getattr_var.as_proxy().node, tx)

                    # same tensor identiy, setattr is a no-op
                    mod_setattr = inspect.getattr_static(obj.module_type, "__setattr__")
                    if (
                        existing_fake_attr is assigning_fake_val
                        and mod_setattr is torch.nn.Module.__setattr__
                    ):
                        return getattr_var

            obj.convert_to_unspecialized(tx)
        # FIXME (tmanlaibaatar) this is utter hack to unblock HuggingFace export
        # Export generally doesn't want to allow mutations on objects directly,
        # but we don't have good way to do this rn. For now, we make it an undefined
        # behaviour and just set attributes directly on the PretrainedConfig object
        # for now.
        elif isinstance(obj, variables.dicts.HFPretrainedConfigVariable) and tx.export:
            if name_var.is_python_constant() and isinstance(
                val, variables.ConstantVariable
            ):
                setattr(
                    obj.obj, name_var.as_python_constant(), val.as_python_constant()
                )
                return ConstantVariable(None)

    def call_delattr(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker,
        name_var: VariableTracker,
    ):
        return self.call_setattr(tx, obj, name_var, variables.DeletedVariable())

    def call_type(self, tx: "InstructionTranslator", obj: VariableTracker):
        try:
            py_type = obj.python_type()
        except NotImplementedError as error:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                str(error),
                case_name="unknown_python_type",
            ) from None

        source = obj.source and TypeSource(obj.source)
        if py_type is torch.Tensor:
            # In some cases torch isn't available in globals
            name = tx.output.install_global_by_id("", torch)
            source = AttrSource(GlobalSource(name), "Tensor")

        return VariableTracker.build(tx, py_type, source)

    def call_reversed(self, tx: "InstructionTranslator", obj: VariableTracker):
        if obj.has_unpack_var_sequence(tx):
            items = list(reversed(obj.unpack_var_sequence(tx)))
            return variables.TupleVariable(items)

    def call_sorted(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker,
        **kwargs: VariableTracker,
    ):
        if obj.has_force_unpack_var_sequence(tx) and not isinstance(
            obj, variables.TensorVariable
        ):
            list_var = variables.ListVariable(
                obj.force_unpack_var_sequence(tx),
                mutation_type=ValueMutationNew(),
            )
            list_var.call_method(tx, "sort", [], kwargs)
            return list_var

    # neg is a constant fold function, so we only get here if constant fold is not valid
    def call_neg(self, tx: "InstructionTranslator", a):
        if isinstance(a, SymNodeVariable):
            return SymNodeVariable.create(
                tx,
                (operator.neg)(a.as_proxy()),
                sym_num=None,
            )
        # None no-ops this handler and lets the driving function proceed
        return None

    def call_format(self, tx: "InstructionTranslator", _format_string, *args, **kwargs):
        format_string = _format_string.as_python_constant()
        return variables.StringFormatVariable.create(format_string, args, kwargs)

    def call_id(self, tx: "InstructionTranslator", *args):
        if len(args) > 0 and isinstance(args[0], variables.NNModuleVariable):
            nn_mod_variable = args[0]
            mod = tx.output.get_submodule(nn_mod_variable.module_key)
            return variables.ConstantVariable.create(id(mod))
        elif len(args) == 1 and isinstance(
            args[0], variables.UserDefinedObjectVariable
        ):
            install_guard(args[0].source.make_guard(GuardBuilder.ID_MATCH))
            constant_result = id(args[0].value)
            return variables.ConstantVariable.create(constant_result)
        elif len(args) == 1 and isinstance(args[0], TensorVariable):
            tensor_variable = args[0]
            return tensor_variable.call_id(tx)
        else:
            unimplemented(f"call_id with args {args}")

    def call_deepcopy(self, tx: "InstructionTranslator", x):
        unimplemented(f"copy.deepcopy {repr(x)}")

    def _comparison_with_tensor(self, tx: "InstructionTranslator", left, right):
        from .builder import wrap_fx_proxy_cls
        from .tensor import supported_tensor_comparison_op_values

        op = self.fn

        if op in [operator.is_, operator.is_not]:
            is_result = (
                isinstance(left, TensorVariable)
                and isinstance(right, TensorVariable)
                and id(extract_fake_example_value(left.as_proxy().node))
                == id(extract_fake_example_value(right.as_proxy().node))
            )
            if op is operator.is_:
                return ConstantVariable.create(is_result)
            else:
                return ConstantVariable.create(not is_result)

        if op not in supported_tensor_comparison_op_values:
            unimplemented(f"{op.__name__}({left}, {right})")
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
                unimplemented(f"{op.__name__}({left}, {right})")
        tensor_cls = left if isinstance(left, TensorVariable) else right
        proxy = tx.output.create_proxy(
            "call_function", op, (left.as_proxy(), right.as_proxy()), {}
        )
        return wrap_fx_proxy_cls(
            type(tensor_cls),  # handle Ndarrays and Tensors
            tx,
            proxy,
        )

    def _comparison_with_symnode(self, tx: "InstructionTranslator", left, right):
        from .tensor import supported_tensor_comparison_op_values

        op = self.fn

        if op not in supported_tensor_comparison_op_values:
            unimplemented(f"{op.__name__}({left}, {right})")

        proxy = tx.output.create_proxy(
            "call_function", op, (left.as_proxy(), right.as_proxy()), {}
        )
        return SymNodeVariable.create(
            tx,
            proxy,
            sym_num=None,
        )

    def call_and_(self, tx: "InstructionTranslator", a, b):
        # Rely on constant_handler
        if isinstance(a, ConstantVariable) and isinstance(b, ConstantVariable):
            return None
        if isinstance(a, (SymNodeVariable, ConstantVariable)) and isinstance(
            b, (SymNodeVariable, ConstantVariable)
        ):
            return SymNodeVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_function", operator.and_, *proxy_args_kwargs([a, b], {})
                ),
                sym_num=None,
            )
        if hasattr(a, "set_items") and hasattr(b, "set_items"):
            return SetVariable(list(a.set_items & b.set_items))
        # None no-ops this handler and lets the driving function proceed

    call_iand = call_and_

    def call_or_(self, tx: "InstructionTranslator", a, b):
        # Rely on constant_handler
        if isinstance(a, ConstantVariable) and isinstance(b, ConstantVariable):
            return None
        if isinstance(a, (SymNodeVariable, ConstantVariable)) and isinstance(
            b, (SymNodeVariable, ConstantVariable)
        ):
            return SymNodeVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_function", operator.or_, *proxy_args_kwargs([a, b], {})
                ),
                sym_num=None,
            )
        if hasattr(a, "set_items") and hasattr(b, "set_items"):
            return SetVariable(list(a.set_items | b.set_items))
        # None no-ops this handler and lets the driving function proceed
        return None

    call_ior = call_or_

    def call_not_(self, tx: "InstructionTranslator", a):
        if isinstance(a, SymNodeVariable):
            return SymNodeVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_function", operator.not_, *proxy_args_kwargs([a], {})
                ),
                sym_num=None,
            )

        # Unwrap the underlying ConstDictVariable
        if isinstance(a, DictView):
            a = a.dv_dict
        if isinstance(a, (ListVariable, ConstDictVariable)):
            return ConstantVariable.create(len(a.items) == 0)

        return None

    def call_contains(
        self, tx: "InstructionTranslator", a: VariableTracker, b: VariableTracker
    ):
        return a.call_method(tx, "__contains__", [b], {})


@contextlib.contextmanager
def dynamo_disable_grad(tx):
    from . import GradModeVariable

    org_value = torch.is_grad_enabled()
    gmv = GradModeVariable.create(tx, False)
    try:
        gmv.enter(tx)
        yield
    finally:
        gmv.exit(tx)
