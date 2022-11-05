import functools
import inspect
import itertools
import logging
import math
import operator
import types
from typing import Dict, List

import numpy as np

import torch
from torch._prims_common import (
    Number,
)
from torch.fx.experimental.symbolic_shapes import sym_int, sym_float

from .. import config, variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented, Unsupported
from ..guards import GuardBuilder
from ..replay_record import DummyModule
from ..source import AttrSource, is_constant_source, TypeSource
from ..utils import (
    check_constant_args,
    check_unspec_python_args,
    istype,
    proxy_args_kwargs,
    specialize_args_kwargs,
)
from .base import MutableLocal, VariableTracker
from .dicts import ConstDictVariable
from .tensor import DynamicShapeVariable, FakeItemVariable, TensorVariable

log = logging.getLogger(__name__)


class BuiltinVariable(VariableTracker):
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
            dict,
            divmod,
            float,
            int,
            len,
            list,
            max,
            min,
            ord,
            pow,
            repr,
            round,
            set,
            str,
            str.format,
            sum,
            tuple,
            type,
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
            operator.sub,
            operator.getitem,
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
        fns.update(x for x in math.__dict__.values() if isinstance(x, type(math.sqrt)))
        return fns

    def can_constant_fold_through(self):
        return self.fn in self._constant_fold_functions()

    @staticmethod
    @functools.lru_cache(None)
    def _fx_graph_functions():
        fns = {
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
            operator.sub,
            operator.getitem,
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
        }
        return fns

    def can_insert_in_graph(self):
        return self.fn in self._fx_graph_functions()

    def __init__(self, fn, **kwargs):
        super(BuiltinVariable, self).__init__(**kwargs)
        self.fn = fn

    def __str__(self):
        if self.fn is None:
            name = "None"
        else:
            name = self.fn.__name__

        return f"{self.__class__.__name__}({name})"

    def python_type(self):
        return type(self.fn)

    def as_python_constant(self):
        return self.fn

    def reconstruct(self, codegen):
        name = self.fn.__name__
        assert self.fn.__module__ == "builtins"
        assert name not in codegen.tx.f_globals, "shadowed global"
        return [codegen.create_load_global(name, add=True)]

    def constant_args(self, *args, **kwargs):
        return check_constant_args(args, kwargs)

    def tensor_args(self, *args, **kwargs):
        return any(
            isinstance(i, variables.TensorVariable)
            for i in itertools.chain(args, kwargs.values())
        ) and not any(
            isinstance(i, variables.GetAttrVariable)
            for i in itertools.chain(args, kwargs.values())
        )

    def unspec_numpy_args(self, *args, **kwargs):
        return all(
            isinstance(
                i,
                (
                    variables.UnspecializedNumpyVariable,
                    variables.UnspecializedPythonVariable,
                    variables.ConstantVariable,
                ),
            )
            for i in itertools.chain(args, kwargs.values())
        ) and any(
            isinstance(x, variables.UnspecializedNumpyVariable)
            for x in itertools.chain(args, kwargs.values())
        )

    def unspec_python_args(self, *args, **kwargs):
        return check_unspec_python_args(args, kwargs)

    @staticmethod
    def unwrap_unspec_args_kwargs(args, kwargs):
        unwrapped_args = []
        unwrapped_kwargs = {}
        for x in args:
            if isinstance(
                x,
                (
                    variables.UnspecializedNumpyVariable,
                    variables.UnspecializedPythonVariable,
                ),
            ):
                unwrapped_args.append(x.raw_value)
            else:
                unwrapped_args.append(x.as_python_constant())
        for k, v in kwargs:
            if isinstance(
                x,
                (
                    variables.UnspecializedNumpyVariable,
                    variables.UnspecializedPythonVariable,
                ),
            ):
                unwrapped_kwargs.update({k: v.raw_value})
            else:
                unwrapped_kwargs.update({k: v.as_python_constant()})
        return unwrapped_args, unwrapped_kwargs

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # print("CALLING BUILTIN", self.fn, args)
        constant_args = check_constant_args(args, kwargs)
        tensor_args = self.tensor_args(*args, **kwargs)
        unspec_python_args = self.unspec_python_args(*args, **kwargs)
        options = VariableTracker.propagate(self, args, kwargs.values())
        has_constant_handler = self.can_constant_fold_through() and (
            constant_args or unspec_python_args
        )
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)

        if (
            self.fn is operator.getitem
            and len(args) == 2
            and isinstance(args[1], variables.TensorVariable)
            and args[1].dtype == torch.bool
            and not config.dynamic_shapes
        ):
            unimplemented("dynamic Tensor.__getitem__(bool[])")

        # args[0] is list and args[1] is unspec
        if self.fn is operator.getitem and not isinstance(
            args[0], variables.TensorVariable
        ):
            tensor_args = False
            args, kwargs = specialize_args_kwargs(tx, args, kwargs)

        # if (
        #     self.can_insert_in_graph()
        #     and tensor_args
        #     and not (
        #         self.fn is operator.getitem
        #         and isinstance(args[0], ConstDictVariable)
        #         and isinstance(args[1], variables.TupleVariable)
        #     )
        # ):

        if (
            self.can_insert_in_graph()
            and tensor_args
            and not (
                self.fn is operator.getitem
                and isinstance(args[0], ConstDictVariable)
                and isinstance(args[1], variables.TensorVariable)
            )
        ):
            try:
                fn = self.fn
                if self.fn is operator.iadd and isinstance(
                    args[0], variables.ConstantVariable
                ):
                    # Work around weird bug in hf_T5
                    fn, args = operator.add, [args[1], args[0]]

                proxy = tx.output.create_proxy(
                    "call_function", fn, *proxy_args_kwargs(args, kwargs), current_tx=tx
                )
                if any([isinstance(arg, FakeItemVariable) for arg in args]):
                    return variables.FakeItemVariable.create(
                        tx,
                        proxy,
                        **options,
                    )
                elif self.unspec_numpy_args(*args, **kwargs):
                    _args, _kwargs = self.unwrap_unspec_args_kwargs(args, kwargs)
                    raw_value = self.fn(*_args, **_kwargs)
                    return variables.UnspecializedNumpyVariable.create(
                        tx,
                        proxy,
                        raw_value=raw_value,
                        **options,
                    )
                elif self.unspec_python_args(*args, **kwargs):
                    _args, _kwargs = self.unwrap_unspec_args_kwargs(args, kwargs)
                    raw_value = self.fn(*_args, **_kwargs)

                    need_unwrap = any(
                        x.need_unwrap
                        for x in itertools.chain(args, kwargs.values())
                        if isinstance(x, variables.UnspecializedPythonVariable)
                    )

                    return variables.UnspecializedPythonVariable.create(
                        tx,
                        proxy,
                        raw_value=raw_value,
                        need_unwrap=need_unwrap,
                        **options,
                    )
                else:
                    # Work around for vision_maskrcnn due to precision difference
                    # specialize the dividend when float divide by tensor
                    if self.fn is operator.truediv and isinstance(
                        args[0], variables.UnspecializedPythonVariable
                    ):
                        args[0] = args[0].convert_to_constant(tx)
                    return variables.TensorVariable.create(tx, proxy, **options)

            except NotImplementedError:
                unimplemented(f"partial tensor op: {self} {args} {kwargs}")

        # Handle cases like int(torch.seed())
        # Also handle sym_float to sym_int cases
        if self.fn in (int, float) and isinstance(args[0], DynamicShapeVariable):
            fn_ = sym_int if self.fn is int else sym_float
            out = TensorVariable.create(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    fn_,
                    (args[0].as_proxy(),),
                    {},
                    current_tx=tx,
                ),
                **options,
            )
            return out


        handler = getattr(self, f"call_{self.fn.__name__}", None)
        if handler:
            try:
                inspect.signature(handler).bind(tx, *args, **kwargs)
            except TypeError as exc:
                if not has_constant_handler:
                    log.warning(
                        f"incorrect arg count {handler} {exc} and no constant handler"
                    )
                handler = None

        if handler:
            try:
                result = handler(tx, *args, **kwargs)
                if result is not None:
                    return result.add_options(options)
            except Unsupported as exc:
                if not has_constant_handler:
                    raise
                # Actually, we will handle this just fine
                exc.remove_from_stats()

        if has_constant_handler:
            args, kwargs = specialize_args_kwargs(tx, args, kwargs)
            # constant fold
            return variables.ConstantVariable(
                self.as_python_constant()(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
                **options,
            )
        if any([isinstance(x, DynamicShapeVariable) for x in args]) or any(
            [isinstance(x, DynamicShapeVariable) for x in kwargs.values()]
        ):
            proxy = tx.output.create_proxy(
                "call_function", self.fn, *proxy_args_kwargs(args, kwargs)
            )
            value = None
            if self.fn == range:
                assert len(kwargs) == 0

                def guard_if_dyn(arg):
                    if isinstance(arg, DynamicShapeVariable):
                        return arg.evaluate_expr(tx.output)
                    return arg

                args = [guard_if_dyn(arg) for arg in args]
                value = self.fn(*args)

            return DynamicShapeVariable.create(tx, proxy, value, **options)
        return super().call_function(tx, args, kwargs)

    def _call_min_max(self, tx, a, b):
        if self.tensor_args(a, b):
            if not isinstance(a, variables.TensorVariable):
                a, b = b, a
            assert isinstance(a, variables.TensorVariable)

            # result of an item call is a scalar convert to a tensor
            if isinstance(a, FakeItemVariable):
                a = variables.TorchVariable(torch.tensor).call_function(tx, [a], {})

            # Dynamic input does not get resolved, rather, gets stored as call_function
            if isinstance(a, DynamicShapeVariable):
                return variables.TensorVariable.create(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        self.fn,
                        *proxy_args_kwargs([a, b], {}),
                        current_tx=tx,
                    ),
                    **VariableTracker.propagate(self, [a, b]),
                )

            # convert min/max to torch ops
            if b.is_python_constant():
                kwargs = {"min": b} if (self.fn is max) else {"max": b}
                result = variables.TorchVariable(torch.clamp).call_function(
                    tx, [a], kwargs
                )
            else:
                fn = {max: torch.maximum, min: torch.minimum}[self.fn]
                result = variables.TorchVariable(fn).call_function(tx, [a, b], {})

            # return unspec if both a, b are unspec or const
            if all(
                isinstance(
                    i,
                    (
                        variables.UnspecializedNumpyVariable,
                        variables.UnspecializedPythonVariable,
                        variables.ConstantVariable,
                    ),
                )
                for i in [a, b]
            ):

                if any([isinstance(val, FakeItemVariable) for val in [a, b]]):
                    return variables.FakeItemVariable.from_tensor_variable(result)

                if b.is_python_constant():
                    raw_b = b.as_python_constant()
                else:
                    raw_b = b.raw_value
                if self.fn is max:
                    raw_res = max(a.raw_value, raw_b)
                else:
                    raw_res = min(a.raw_value, raw_b)

                if isinstance(raw_res, np.number):
                    return variables.UnspecializedNumpyVariable.from_tensor_variable(
                        result, raw_res
                    )
                else:
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
        elif isinstance(a, variables.ConstantVariable) and isinstance(
            b, variables.ConstantVariable
        ):
            if self.fn is max:
                return variables.ConstantVariable(max(a.value, b.value))
            else:
                return variables.ConstantVariable(min(a.value, b.value))
        elif isinstance(a, DynamicShapeVariable) or isinstance(b, DynamicShapeVariable):
            proxy = tx.output.create_proxy(
                "call_function", self.fn, *proxy_args_kwargs([a, b], {})
            )
            return DynamicShapeVariable.create(tx, proxy, None)
        else:

            unimplemented(f"unsupported min / max over args {str(a)}, {str(b)}")

    call_min = _call_min_max
    call_max = _call_min_max

    def call_range(self, tx, *args, **kwargs):
        if self.unspec_python_args(*args, **kwargs) or self.constant_args(
            *args, **kwargs
        ):
            args, kwargs = specialize_args_kwargs(tx, args, kwargs)
            return variables.RangeVariable(
                value=range(
                    *[x.value for x in args],
                    **{k: v.value for k, v in kwargs.items()},
                ),
            )

    def call_slice(self, tx, *args):
        return variables.SliceVariable(args)

    def _call_iter_tuple_list(self, tx, obj=None):
        cls = variables.BaseListVariable.cls_for(self.fn)
        if obj is None:
            return cls(
                [],
                mutable_local=MutableLocal(),
            )
        elif obj.has_unpack_var_sequence(tx):
            guards = set()
            if obj.source and not is_constant_source(obj.source):
                guards.add(obj.source.make_guard(GuardBuilder.LIST_LENGTH))
            return cls(
                list(obj.unpack_var_sequence(tx)),
                mutable_local=MutableLocal(),
                guards=guards,
            ).add_options(self, obj)

    call_iter = _call_iter_tuple_list
    call_tuple = _call_iter_tuple_list
    call_list = _call_iter_tuple_list

    def call_dict(self, tx, arg):
        if isinstance(arg, variables.ConstDictVariable):
            return arg.clone(mutable_local=MutableLocal())

    def call_zip(self, tx, *args):
        options = VariableTracker.propagate(self, args)
        if all(x.has_unpack_var_sequence(tx) for x in args):
            items = [
                variables.TupleVariable(list(item), **options)
                for item in zip(*[arg.unpack_var_sequence(tx) for arg in args])
            ]
            return variables.TupleVariable(items, **options)

    def call_enumerate(self, tx, *args):
        options = VariableTracker.propagate(self, args)
        if len(args) == 1:
            start = 0
        else:
            assert len(args) == 2
            assert isinstance(args[1], variables.ConstantVariable)
            start = args[1].as_python_constant()
        if args[0].has_unpack_var_sequence(tx):
            items = [
                variables.TupleVariable(
                    [variables.ConstantVariable(idx, **options), var],
                    **options,
                )
                for idx, var in enumerate(args[0].unpack_var_sequence(tx), start)
            ]
            return variables.TupleVariable(items, **options)

    def _call_operator_builtin(self, tx, op_name: str, *args, **kwargs):
        options = VariableTracker.propagate(self, args)
        fn_ = getattr(operator, op_name)

        # This should really be refactored.
        # builtins like + and - can return a ton of different types,
        # depending on the types of the inputs.
        # Right now, TensorVariable.create() handles a subset of these cases,
        # but not all of them.
        # For example, different cases that we care about include
        # when the output type is a:
        # - ConstantVariable (handled by ConstantVariable.call_method)
        # - DynamicShapesVariable (handled by TensorVariable.create)
        # - TensorVariable (handled by TensorVariable.create)
        if all(isinstance(x, variables.ConstantVariable) for x in args):
            return args[0].call_method(tx, f'__{op_name}__', args[1:], kwargs)

        out = TensorVariable.create(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                fn_,
                *proxy_args_kwargs(args, kwargs),
                current_tx=tx,
            ),
            **options,
        )
        return out

    def call_mul(self, tx, a, b):
        if isinstance(
            a, (variables.ListVariable, variables.TupleVariable)
        ) and isinstance(b, variables.ConstantVariable):
            return a.__class__(
                items=a.items * b.as_python_constant(), mutable_local=MutableLocal()
            ).add_options(self, a, b)
        elif isinstance(
            b, (variables.ListVariable, variables.TupleVariable)
        ) and isinstance(a, variables.ConstantVariable):
            return b.__class__(
                items=b.items * a.as_python_constant(), mutable_local=MutableLocal()
            ).add_options(self, a, b)
        else:
            return self._call_operator_builtin(tx, "mul", a, b)

    def call_len(self, tx, *args, **kwargs):
        return args[0].call_method(tx, "__len__", args[1:], kwargs)

    def call_add(self, tx, *args, **kwargs):
        # Important: we want to trace operator.add(x, y) into the graph,
        # not x.__add__(y).
        # operator.add will automatically handle NotImplemented argument swizzling.
        return self._call_operator_builtin(tx, "add", *args, *kwargs)

    def call_sub(self, tx, *args, **kwargs):
        return self._call_operator_builtin(tx, "sub", *args, *kwargs)

    def call_truediv(self, tx, *args, **kwargs):
        return self._call_operator_builtin(tx, "truediv", *args, *kwargs)

    def call_floordiv(self, tx, *args, **kwargs):
        return self._call_operator_builtin(tx, "floordiv", *args, *kwargs)

    def call_iadd(self, tx, *args, **kwargs):
        return args[0].call_method(tx, "__iadd__", args[1:], kwargs)

    def call_getitem(self, tx, *args, **kwargs):
        if self.unspec_python_args(*args, **kwargs):
            args, kwargs = specialize_args_kwargs(tx, args, kwargs)
        return args[0].call_method(tx, "__getitem__", args[1:], kwargs)

    def call_isinstance(self, tx, arg, isinstance_type):
        arg_type = arg.python_type()
        isinstance_type = isinstance_type.as_python_constant()

        if isinstance(arg, variables.TensorVariable) and arg.dtype is not None:
            return variables.ConstantVariable(arg.call_isinstance(isinstance_type))
        # UserDefinedObject with C extensions can have torch.Tensor attributes,
        # so break graph.
        if isinstance(arg, variables.UserDefinedObjectVariable) and isinstance(
            arg.value, types.MemberDescriptorType
        ):
            unimplemented(
                f"isinstance called on UserDefinedClass {arg} {isinstance_type}"
            )
        try:
            val = issubclass(arg_type, isinstance_type)
        except TypeError:
            val = arg_type is isinstance_type
        return variables.ConstantVariable(val)

    def call_super(self, tx, a, b):
        return variables.SuperVariable(a, b)

    def call_next(self, tx, arg):
        if isinstance(arg, variables.ListIteratorVariable):
            val, next_iter = arg.next_variables()
            tx.replace_all(arg, next_iter)
            return val
        elif isinstance(arg, variables.BaseListVariable):
            return arg.items[0].add_options(self, arg)

    def call_hasattr(self, tx, obj, attr):
        if attr.is_python_constant():
            name = attr.as_python_constant()
            return obj.call_hasattr(tx, name).add_options(self, obj, attr)

    def call_map(self, tx, fn, seq):
        if seq.has_unpack_var_sequence(tx):
            items = [fn.call_function(tx, [x], {}) for x in seq.unpack_var_sequence(tx)]
            return variables.TupleVariable(items).add_options(self, fn, seq)

    def call_sum(self, tx, seq, **kwargs):
        # Special case for sum on tuple of floats and ints
        if (
            isinstance(seq, (variables.ListVariable, variables.TupleVariable))
            and all(
                [
                    isinstance(x, variables.ConstantVariable)
                    and isinstance(x.value, (int, float))
                    for x in seq.items
                ]
            )
            and not kwargs
        ):
            new_list = [x.value for x in seq.items]
            return variables.ConstantVariable(sum(new_list))
        if seq.has_unpack_var_sequence(tx):
            start = kwargs.pop(
                "start", variables.ConstantVariable(0)
            ).as_python_constant()
            assert not kwargs
            items = seq.unpack_var_sequence(tx)[start:]
            return BuiltinVariable(functools.reduce).call_function(
                tx,
                [
                    BuiltinVariable(operator.add),
                    variables.TupleVariable(items),
                    variables.ConstantVariable(0).add_options(self, seq),
                ],
                {},
            )

    def call_reduce(self, tx, function, iterable, initializer=None):
        if iterable.has_unpack_var_sequence(tx):
            items = iterable.unpack_var_sequence(tx)
            if initializer is None:
                value, items = items[0], items[1:]
            else:
                value = initializer
            for element in items:
                value = function.call_function(tx, [value, element], {})
            return value

    def call_getattr(
        self, tx, obj: VariableTracker, name_var: VariableTracker, default=None
    ):
        from . import (
            ConstantVariable,
            GetAttrVariable,
            PythonModuleVariable,
            TorchVariable,
            UserFunctionVariable,
        )
        from .builder import VariableBuilder

        options = VariableTracker.propagate(self, obj, name_var)
        guards = options["guards"]
        name = name_var.as_python_constant()

        if not name_var.is_python_constant():
            unimplemented("non-const getattr() name")

        if tx.output.side_effects.is_attribute_mutation(obj):
            try:
                # re-read a pending side effect?
                return tx.output.side_effects.load_attr(obj, name).add_options(options)
            except KeyError:
                pass

        if default is not None:
            hasattr_var = self.call_hasattr(tx, obj, name_var)
            guards.update(hasattr_var.guards)
            assert hasattr_var.as_python_constant() in (True, False)
            if not hasattr_var.as_python_constant():
                return default.add_guards(guards)

        if obj.source:
            source = AttrSource(obj.source, name)
            options["source"] = source
        else:
            source = None

        if isinstance(obj, variables.NNModuleVariable):
            return obj.var_getattr(tx, name).add_options(options)
        elif isinstance(obj, variables.TensorVariable) and name == "grad":
            if source:
                # We are going to be raising this tensor as grapharg. So, ensure
                # that we have real grad value instead of fake tensor value.
                # Walk through the inputs of the subgraph and find if we already
                # have the original tensor stored in the graphargs.
                for grapharg in tx.output.graphargs:
                    if grapharg.source == source.base:
                        example_value = grapharg.example.grad
                        return VariableBuilder(tx, source)(example_value).add_options(
                            options
                        )
                unimplemented("tensor grad")
            else:
                unimplemented("tensor grad")
        elif isinstance(
            obj,
            (
                variables.TensorVariable,
                variables.NamedTupleVariable,
                variables.ConstantVariable,
                variables.UserDefinedClassVariable,
                variables.UserDefinedObjectVariable,
            ),
        ):
            try:
                return (
                    obj.var_getattr(tx, name).clone(source=source).add_options(options)
                )
            except NotImplementedError:
                return GetAttrVariable(obj, name, **options)
        elif isinstance(obj, TorchVariable):
            member = getattr(obj.value, name)
            if is_allowed(member):
                return TorchVariable(member, **options)
            elif ConstantVariable.is_literal(member):
                return ConstantVariable(member, **options)
            else:
                return VariableBuilder(tx, source)(member).add_guards(guards)
        elif isinstance(obj, (PythonModuleVariable, DummyModule)):
            member = obj.value.__dict__[name]

            if config.replay_record_enabled:
                tx.exec_recorder.record_module_access(obj.value, name, member)

            return VariableBuilder(tx, source)(member).add_guards(guards)
        elif istype(obj, UserFunctionVariable) and name in ("__name__", "__module__"):
            return ConstantVariable(
                getattr(obj.fn, name), **VariableTracker.propagate(obj)
            )
        else:
            try:
                return (
                    obj.var_getattr(tx, name).clone(source=source).add_options(options)
                )
            except NotImplementedError:
                return GetAttrVariable(obj, name, **options)

    def call_setattr(
        self, tx, obj: VariableTracker, name_var: VariableTracker, val: VariableTracker
    ):
        if isinstance(obj, (variables.BlackHoleVariable, variables.DataClassVariable)):
            return obj.call_method(tx, "__setattr__", [name_var, val], {})
        elif (
            tx.output.side_effects.is_attribute_mutation(obj)
            and name_var.is_python_constant()
        ):
            tx.output.side_effects.store_attr(obj, name_var.as_python_constant(), val)
            return val.add_options(self, obj, name_var)
        elif isinstance(obj, variables.UserDefinedObjectVariable):
            unimplemented(
                f"setattr(UserDefinedObjectVariable) {type(obj.value).__setattr__}"
            )
        elif isinstance(obj, variables.NNModuleVariable):
            obj.convert_to_unspecialized(tx)

    def call_type(self, tx, obj: VariableTracker):
        from .builder import VariableBuilder

        try:
            py_type = obj.python_type()
        except NotImplementedError:
            py_type = None

        if istype(obj, variables.TupleVariable):
            return BuiltinVariable(py_type).add_options(self, obj)

        if py_type is not None and obj.source:
            return VariableBuilder(tx, TypeSource(obj.source))(py_type).add_options(
                self, obj
            )

        unimplemented(f"type({obj})")

    def call_reversed(self, tx, obj: VariableTracker):
        if obj.has_unpack_var_sequence(tx):
            items = list(reversed(obj.unpack_var_sequence(tx)))
            return variables.TupleVariable(
                items, **VariableTracker.propagate(self, obj)
            )

    def call_chain(self, tx, *args):
        if all(obj.has_unpack_var_sequence(tx) for obj in args):
            items = []
            for obj in args:
                items.extend(obj.unpack_var_sequence(tx))
            return variables.TupleVariable(
                items, **VariableTracker.propagate(self, *args)
            )

    def call_islice(self, tx, iterable, *args):
        if iterable.has_unpack_var_sequence(tx) and all(
            x.is_python_constant() for x in args
        ):
            const_args = [x.as_python_constant() for x in args]
            items = iterable.unpack_var_sequence(tx)
            items = list(itertools.islice(items, *const_args))
            return variables.TupleVariable(
                items, **VariableTracker.propagate(self, iterable, *args)
            )

    def call_id(self, tx, *args):
        if len(args) > 0 and isinstance(args[0], variables.NNModuleVariable):
            nn_mod_variable = args[0]
            mod = tx.output.get_submodule(nn_mod_variable.module_key)
            return variables.ConstantVariable(id(mod))
        else:
            unimplemented(f"call_id with args {args}")
