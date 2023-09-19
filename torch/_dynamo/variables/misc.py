import collections
import functools
import inspect
import itertools
import sys
import types
from typing import Dict, List

import torch._C
import torch._numpy as tnp
from .. import config, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..source import AttrSource, ODictGetItemSource
from ..utils import check_constant_args, identity, proxy_args_kwargs
from .base import MutableLocal, VariableTracker
from .dicts import DefaultDictVariable
from .functions import (
    NestedUserFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)
from .user_defined import UserDefinedObjectVariable


class SuperVariable(VariableTracker):
    def __init__(self, typevar, objvar=None, specialized=False, **kwargs):
        super().__init__(**kwargs)
        self.typevar = typevar
        self.objvar = objvar
        self.specialized = specialized  # directly get attr from self.typevar if true

    def reconstruct(self, codegen):
        codegen(variables.BuiltinVariable(super))
        codegen(self.typevar)
        if self.objvar is not None:
            codegen(self.objvar)
            return create_call_function(2, True)
        else:
            return create_call_function(1, True)

    def const_getattr(self, tx, name):
        assert self.objvar, "1-arg super not implemented"
        if self.specialized:
            return getattr(self.typevar.as_python_constant(), name)
        search_type = self.typevar.as_python_constant()

        # We default to the python type of the object. However, if this is
        # a `type` or subclass of `type`, then the original object represents
        # the user defined type.
        type_to_use = self.objvar.python_type()
        if issubclass(type_to_use, type):
            type_to_use = self.objvar.value

        # TODO(jansel): there is a small chance this could trigger user code, prevent that
        return getattr(super(search_type, type_to_use), name)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(
            self, args, kwargs.values(), self.objvar, self.typevar
        )
        inner_fn = self.const_getattr(self, name)
        source = None if self.source is None else AttrSource(self.source, name)
        if inner_fn is object.__init__:
            return LambdaVariable(identity, **options)
        elif inner_fn is torch.nn.Module.__init__:
            objvar = self.objvar
            from ..side_effects import AttributeMutationNew

            if (
                isinstance(objvar, variables.UserDefinedObjectVariable)
                and isinstance(objvar.mutable_local, AttributeMutationNew)
                and not (args or kwargs)
            ):
                tx.output.guards.update(options.get("guards", set()))
                tx.output.side_effects.store_attr(
                    objvar, "__call_nn_module_init", variables.ConstantVariable(True)
                )
                return variables.ConstantVariable(None)
            else:
                unimplemented("super() nn.Module.__init__")
        elif isinstance(inner_fn, types.FunctionType):
            return variables.UserFunctionVariable(
                inner_fn, source=source, **options
            ).call_function(tx, [self.objvar] + args, kwargs)
        elif isinstance(inner_fn, types.MethodType):
            return variables.UserMethodVariable(
                inner_fn.__func__, self.objvar, source=source, **options
            ).call_function(tx, args, kwargs)
        elif (
            inner_fn is collections.OrderedDict.__getitem__
            and isinstance(self.objvar, variables.UserDefinedObjectVariable)
            and self.objvar.source
            and len(args) == 1
            and len(kwargs) == 0
            and args[0].is_python_constant()
        ):
            from .builder import VariableBuilder

            key = args[0].as_python_constant()
            return VariableBuilder(tx, ODictGetItemSource(self.objvar.source, key))(
                collections.OrderedDict.__getitem__(self.objvar.value, key)
            )
        elif (
            inner_fn in (collections.OrderedDict.__setitem__, object.__setattr__)
            and isinstance(self.objvar, variables.CustomizedDictVariable)
            and args
            and variables.ConstDictVariable.is_valid_key(args[0])
            and self.objvar.mutable_local
        ):
            assert not kwargs and len(args) == 2
            k = variables.ConstDictVariable.get_key(args[0])

            newval = collections.OrderedDict(self.objvar.items)
            newval[k] = args[1]

            new_rec_contains = self.objvar.recursively_contains.union(
                args[1].recursively_contains
            )
            if args[1].mutable_local is not None:
                new_rec_contains.add(args[1].mutable_local)

            return tx.replace_all(
                self.objvar,
                self.objvar.modifed(newval, new_rec_contains, **options),
            )
        else:
            unimplemented(f"non-function or method super: {inner_fn}")


class UnknownVariable(VariableTracker):
    """
    It could be anything!
    """


class DelayGraphBreakVariable(UnknownVariable):
    """
    Used to insert a dummy variable in the stack to do the graph break at CALL_FUNCTION.
    """


class ComptimeVariable(VariableTracker):
    """
    This variable is special, it lets you execute arbitrary code at
    Dynamo compile time
    """

    def reconstruct(self, codegen):
        raise NotImplementedError("comptime is special form")

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        from ..comptime import comptime

        # To support the comptime.print_graph convenience accessors
        from .functions import UserFunctionVariable

        return UserFunctionVariable(
            getattr(comptime, name), source=AttrSource(self.source, name)
        )

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from ..comptime import ComptimeContext

        # TODO: support an expression form as well

        assert not kwargs
        assert len(args) == 1
        fn = args[0]
        if isinstance(fn, UserFunctionVariable):
            fn.get_function()(ComptimeContext(tx))
        elif isinstance(fn, NestedUserFunctionVariable):
            # We have to manually bind the freevars ourselves
            code = fn.get_code()
            assert not fn.closure, (
                "comptime function must not have free variables, "
                f"but these variables were free: {code.co_freevars}"
            )
            func = types.FunctionType(
                code,
                fn.f_globals,
                fn.fn_name.as_python_constant(),
                tuple(fn.defaults.items) if fn.defaults else None,
                # We could automatically promote free variables into
                # ComptimeVar but this is confusing if you access
                # a free variable that we actually DO have the runtime
                # value for
                # tuple(make_cell(ComptimeVar(i)) for i in fn.closure.items)
                tuple(),
            )
            func(ComptimeContext(tx))
        else:
            raise RuntimeError(f"unsupported argument to comptime: {type(fn)}")

        return variables.ConstantVariable(None)


class ClosureVariable(UnknownVariable):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def reconstruct(self, codegen):
        return [codegen.create_load_closure(self.name)]


# closure variable created by an inlined function
class InlinedClosureVariable(UnknownVariable):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def reconstruct(self, codegen):
        return [codegen.create_load_closure(self.name)]


class NewCellVariable(VariableTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NewGlobalVariable(VariableTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class InspectSignatureVariable(VariableTracker):
    """represents inspect.signature(...)"""

    @staticmethod
    def create(callable, **kwargs):
        if kwargs:
            unimplemented(f"inspect.signature with {kwargs}")
        return InspectSignatureVariable(callable)

    def __init__(self, inspected, **kwargs):
        super().__init__(**kwargs)
        self.inspected = inspected


def produce_trampoline_autograd_fwd(fn_cls):
    def trampoline_autograd_fwd(*args, **kwargs):
        return fn_cls.forward(*args, **kwargs)

    trampoline_autograd_fwd._origin = produce_trampoline_autograd_fwd
    return trampoline_autograd_fwd


def produce_trampoline_autograd_bwd(fn_cls):
    def trampoline_autograd_bwd(*args, **kwargs):
        return fn_cls.backward(*args, **kwargs)

    trampoline_autograd_bwd._origin = produce_trampoline_autograd_bwd
    return trampoline_autograd_bwd


def produce_trampoline_autograd_apply(fn_cls):
    def trampoline_autograd_apply(*args, **kwargs):
        return fn_cls.apply(*args, **kwargs)

    trampoline_autograd_apply._origin = produce_trampoline_autograd_apply
    return trampoline_autograd_apply


class AutogradFunctionVariable(VariableTracker):
    """represents a torch.autograd.Function subclass"""

    def __init__(self, fn_cls, **kwargs):
        super().__init__(**kwargs)
        self.fn_cls = fn_cls

    def call_apply(self, tx, args, kwargs):
        requires_grad = False

        def visit(node):
            nonlocal requires_grad
            if isinstance(node, variables.TensorVariable):
                if node.requires_grad is not False:
                    requires_grad = True
            if isinstance(node, variables.NNModuleVariable):
                if node.is_training(tx):
                    requires_grad = True
            return node

        VariableTracker.apply(visit, (args, kwargs))

        ctx = AutogradFunctionContextVariable.create(tx)
        args = [ctx, *args]

        if (
            requires_grad
            and torch.is_grad_enabled()
            and config.capture_autograd_function
        ):
            # Note - this is the same check used in autograd/function.py, except inverted.
            # If we want to support functorch transforms here, we will need to enable this.
            if (
                self.fn_cls.setup_context
                != torch.autograd.function._SingleLevelFunction.setup_context
            ):
                unimplemented(
                    "NYI - autograd.Function with custom setup_context method"
                )

            vjp_fn = self.fn_cls.vjp  # type: ignore[attr-defined]
            if vjp_fn is not torch.autograd.Function.vjp:
                unimplemented("NYI - User defind vjp")

            jvp_fn = self.fn_cls.jvp  # type: ignore[attr-defined]
            if jvp_fn is not torch.autograd.Function.jvp:
                unimplemented("NYI - User defind jvp")

            from .higher_order_ops import (
                safe_or_raise_always_restore,
                TorchHigherOrderOperatorVariable,
            )

            trampoline_autograd_apply = produce_trampoline_autograd_apply(self.fn_cls)
            trampoline_autograd_fwd = produce_trampoline_autograd_fwd(self.fn_cls)
            trampoline_autograd_bwd = produce_trampoline_autograd_bwd(self.fn_cls)

            # NOTE [On Tracing autograd.Function w/ grad]
            # The complex system described here revolves around the soundness evaluation of an autograd.Function in
            # PyTorch. The system follows a well-defined strategy for tracing, which involves three key steps: tracing
            # forward, tracing backward, and if both are sound the potential recording of an "apply" operation into the
            # graph.We trace forward, and evaluate soundness. Soundness, in this context, refers to the absence of side
            # effects, the avoidance of lifting new arguments into the trace, the production of a single tensor output,
            # and a limited input scope confined to contexts, tensors, and constants. If the forward trace is sound,
            # we install any guards accumulated from tracing. If not, we graph break. We trace backward, and evaluate
            # for soundness, same as forward, except with more strictness. We enable a strict mode on the tx, and
            # reject certain ops when running under this strict mode. If the backward trace is sound, we discard the
            # trace by restoring. Otherwise, we raise.

            # if both the forward and backward traces are sound, we write the autograd function’s apply into the graph.

            # For tracing forward and backward, we use UserFunctionVariable. Although it does not directly contribute
            # to soundness evaluation, it plus a  GlobalSource makes sure we can produce valid guards,
            # and that we can inline properly here. Inlining is required in order to be able to ensure that the
            # soundness evaluation works as described above.
            graph_checkpoint, checkpoint = tx.output.graph, tx.copy_graphstate()

            module_source = AttrSource(
                tx.import_source(self.fn_cls.__module__), self.fn_cls.__name__
            )
            higher_order_autograd_fn = TorchHigherOrderOperatorVariable.make(
                trampoline_autograd_fwd, source=AttrSource(module_source, "forward")
            )
            speculated_fwd_result = higher_order_autograd_fn.call_function(
                tx, args, kwargs
            )

            bwd_args = [ctx, speculated_fwd_result]
            safe_or_raise_always_restore(
                tx,
                graph_checkpoint,
                checkpoint,
                TorchHigherOrderOperatorVariable.make(
                    trampoline_autograd_bwd,
                    source=AttrSource(module_source, "backward"),
                ),
                bwd_args,
            )
            # If fwd and backward are sound, we want apply in the graph.
            # And we don't want backwards for the obvious reasons.
            args = args[1:]
            return TorchHigherOrderOperatorVariable.make(
                trampoline_autograd_apply
            ).call_function(tx, args, kwargs)

        options = VariableTracker.propagate(self, args, kwargs.values())
        options["source"] = AttrSource(AttrSource(self.source, "__class__"), "forward")
        fn = self.fn_cls.forward
        if isinstance(fn, types.FunctionType):
            return variables.UserFunctionVariable(fn, **options).call_function(
                tx, args, kwargs
            )
        elif isinstance(fn, types.MethodType):
            return variables.UserMethodVariable(
                fn.__func__, variables.UserDefinedClassVariable(self.fn_cls), **options
            ).call_function(tx, args, kwargs)
        else:
            unimplemented(
                f"non-function or method in subclass of torch.autograd.Function: {fn}"
            )

    def call_function(self, tx, args, kwargs):
        options = VariableTracker.propagate(self, args, kwargs.values())
        return AutogradFunctionVariable(self.fn_cls, source=self.source, **options)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ):
        if name == "backward":
            with tx.strict_translation_mode():
                if isinstance(self.fn_cls.backward, types.FunctionType):
                    backward = UserFunctionVariable(self.fn_cls.backward)
                elif isinstance(self.fn_cls.backward, types.MethodType):
                    backward = UserMethodVariable(
                        self.fn_cls.backward.__func__,
                        variables.UserDefinedClassVariable(self.fn_cls),
                    )
                    args = [backward.obj] + args
                else:
                    unimplemented(
                        f"backward is a non-function or method: {self.fn_cls.backward}"
                    )

                return tx.inline_call(tx, backward, args, kwargs)

        elif name == "forward":
            if isinstance(self.fn_cls.forward, types.FunctionType):
                forward = UserFunctionVariable(self.fn_cls.forward)
            elif isinstance(self.fn_cls.forward, types.MethodType):
                forward = UserMethodVariable(
                    self.fn_cls.forward.__func__,
                    variables.UserDefinedClassVariable(self.fn_cls),
                )
                args = [forward.obj] + args
            else:
                unimplemented(
                    f"forward is a non-function or method: {self.fn_cls.forward}"
                )

            return tx.inline_call(tx, forward, args, kwargs)

        else:
            unimplemented(f"Unsupported method: {name}")


class AutogradFunctionContextVariable(UserDefinedObjectVariable):
    """
    Tracks an autograd.Function() context using mutation tracking in side_effects.py
    """

    def __init__(self, value, value_type=None, inference=False, **kwargs):
        saved_tensors = kwargs.pop("_saved_tensors", [])
        super().__init__(value=value, value_type=value_type, **kwargs)
        self._saved_tensors = saved_tensors
        self.inference = inference

    @staticmethod
    def create(tx):
        out = tx.output.side_effects.track_object_new(
            None,
            torch.autograd.function.FunctionCtx,
            functools.partial(AutogradFunctionContextVariable, inference=True),
            {},
        )
        proxy = tx.output.create_proxy(
            "call_function", torch.autograd.function.FunctionCtx, tuple(), {}
        )
        proxy.node.meta["example_value"] = out.value

        out.proxy = proxy
        return out

    def as_proxy(self):
        return self.proxy

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name != "save_for_backward":
            unimplemented(f"autograd.Function context method: {name}")

        if not self.inference:
            assert self.source and not kwargs
            tx.output.side_effects.track_save_for_backward(self, args)

        options = VariableTracker.propagate(self, args, kwargs.values())
        if not hasattr(self, "_saved_tensors"):
            self._saved_tensors = []
        for arg in args:
            # as_proxy can return constant values or other non proxy values
            if isinstance(arg.as_proxy(), torch.fx.Proxy):
                arg.as_proxy().node.meta["saved_tensor_marked"] = True
            self._saved_tensors.append(arg)
        return variables.ConstantVariable(None, **options)

    def var_getattr(self, tx, name):
        if name == "save_for_backward":
            return LambdaVariable(
                lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
            ).add_options(self)
        if name == "saved_tensors":
            return variables.TupleVariable(list(self._saved_tensors))
        return super().var_getattr(tx, name)


class LambdaVariable(VariableTracker):
    def __init__(self, fn, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        return self.fn(*args, **kwargs).add_options(self)


class GetAttrVariable(VariableTracker):
    def __init__(self, obj, name, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(obj, VariableTracker)
        assert isinstance(name, str)
        self.obj = obj
        self.name = name

    def __str__(self):
        return f"{self.__class__.__name__}({self.obj}, {self.name})"

    @staticmethod
    def create_getattr_proxy(base_proxy: torch.fx.Proxy, attr):
        return getattr(base_proxy, attr)

    def as_proxy(self):
        return GetAttrVariable.create_getattr_proxy(self.obj.as_proxy(), self.name)

    def const_getattr(self, tx, name):
        if not isinstance(self.obj, variables.NNModuleVariable):
            raise NotImplementedError()
        step1 = tx.output.get_submodule(self.obj.module_key)
        if self.name not in step1.__dict__:
            raise NotImplementedError()
        step2 = inspect.getattr_static(step1, self.name)
        if name not in step2.__dict__:
            raise NotImplementedError()
        return inspect.getattr_static(step2, name)

    def reconstruct(self, codegen):
        codegen(self.obj)
        return codegen.create_load_attrs(self.name)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        # This variable is True when it corresponds to user code such as
        #
        #   super().__torch_function__(...)
        #
        # and the super().__torch_function__ attribute resolves
        # to torch.Tensor.__torch_function__.
        is_original_tensor_torch_function = (
            self.name == "__torch_function__"
            and isinstance(self.obj, SuperVariable)
            # for now, only support one level of inheritance
            and len(self.obj.objvar.value.__mro__) > 1
            and self.obj.objvar.value.__mro__[1] == torch.Tensor
        )
        if is_original_tensor_torch_function:
            # Instead of tracing inside torch.Tensor.__torch_function__,
            # record the `call_function` or `call_method` call into the graph.
            from . import TorchVariable

            original_torch_or_getattr_variable = args[0]
            new_args = args[2].items
            new_kwargs = args[3].items
            options = VariableTracker.propagate(self, new_args, new_kwargs.values())
            # Disable __torch_function__ here to prevent the clone of the
            # example tensor from going into the override.
            with torch._C.DisableTorchFunctionSubclass():
                if isinstance(args[0], TorchVariable):
                    return wrap_fx_proxy(
                        tx=tx,
                        proxy=tx.output.create_proxy(
                            "call_function",
                            original_torch_or_getattr_variable.value,
                            *proxy_args_kwargs(new_args, new_kwargs),
                        ),
                        **options,
                    )
                elif isinstance(args[0], GetAttrVariable):
                    return wrap_fx_proxy(
                        tx=tx,
                        proxy=tx.output.create_proxy(
                            "call_method",
                            original_torch_or_getattr_variable.name,
                            *proxy_args_kwargs(new_args, new_kwargs),
                        ),
                        **options,
                    )
                else:
                    unimplemented(
                        f"GetAttrVariable.call_function original __torch_function__ {args}"
                    )

        if isinstance(self.obj, AutogradFunctionVariable) and self.name == "apply":
            return self.obj.call_apply(tx, args, kwargs).add_options(self)
        # calling parent class‘s non classmethod from child class
        # https://github.com/pytorch/pytorch/issues/90558
        elif (
            isinstance(self.obj, variables.UserDefinedClassVariable)
            and len(args) > 0
            and issubclass(args[0].python_type(), self.obj.value)
        ):
            return SuperVariable(self.obj, args[0], True).call_method(
                tx, self.name, args[1:], kwargs
            )
        return self.obj.call_method(tx, self.name, args, kwargs).add_options(self)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if (
            name == "__len__"
            and isinstance(self.obj, InspectSignatureVariable)
            and self.name == "parameters"
        ):
            return variables.ConstantVariable(
                self.obj.inspected.num_parameters(),
                **VariableTracker.propagate(self, self.obj, self.obj.inspected),
            )
        return super().call_method(tx, name, args, kwargs)


class PythonModuleVariable(VariableTracker):
    def __init__(self, value: types.ModuleType, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def python_type(self):
        return types.ModuleType


class SkipFilesVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    @staticmethod
    @functools.lru_cache(None)
    def fold_through_function_to_wrapper():
        return {
            collections.namedtuple: variables.UserDefinedClassVariable,
        }

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .builtin import BuiltinVariable

        options = VariableTracker.propagate(self, args, kwargs.values())

        if inspect.getattr_static(self.value, "_torchdynamo_disable", False):
            unimplemented(f"call torch._dynamo.disable() wrapped function {self.value}")
        # Allowlist a few popular classes(e.g, collections.OrderedDict) calls in skip files.
        elif self.value is collections.OrderedDict and (
            len(args) == 0
            or len(args) == 1
            and BuiltinVariable.is_supported_call_dict_arg(tx, args[0])
        ):
            if len(args) == 0:
                args = dict(kwargs) if kwargs else None
            else:
                args = args[0]

            return BuiltinVariable.call_dict_helper(
                tx,
                collections.OrderedDict,
                args,
                **options,
            )
        elif (
            self.value is collections.defaultdict
            and len(args) <= 1
            and DefaultDictVariable.is_supported_arg(args[0])
        ):
            return DefaultDictVariable(
                {},
                collections.defaultdict,
                args[0],
                mutable_local=MutableLocal(),
                **options,
            )
        # Fold through the functions(e.g, collections.namedtuple)
        # that inputs & outputs are all python constants
        elif (
            self.value in self.fold_through_function_to_wrapper().keys()
            and check_constant_args(args, kwargs)
        ):
            value = self.value(
                *[x.as_python_constant() for x in args],
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
            return self.fold_through_function_to_wrapper().get(self.value)(
                value, mutable_local=MutableLocal(), **options
            )
        elif (
            self.value is itertools.product
            and not kwargs
            and all(arg.has_unpack_var_sequence(tx) for arg in args)
        ):
            seqs = [arg.unpack_var_sequence(tx) for arg in args]
            items = []
            for item in itertools.product(*seqs):
                items.append(variables.TupleVariable(list(item), **options))
            return variables.ListIteratorVariable(
                items, mutable_local=MutableLocal(), **options
            )
        elif (
            self.value is itertools.chain
            and not kwargs
            and all(arg.has_unpack_var_sequence(tx) for arg in args)
        ):
            seqs = [arg.unpack_var_sequence(tx) for arg in args]
            items = []
            for item in itertools.chain(*seqs):
                items.append(item)
            return variables.ListIteratorVariable(
                items, mutable_local=MutableLocal(), **options
            )
        elif self.value is itertools.accumulate and not kwargs:
            from .builtin import BuiltinVariable

            if len(args) == 1 and args[0].has_unpack_var_sequence(tx):
                seq = args[0].unpack_var_sequence(tx)

                def func(tx, item, acc):
                    return BuiltinVariable(sum).call_function(tx, [item, acc], {})

            elif (
                len(args) == 2
                and args[0].has_unpack_var_sequence(tx)
                and args[1].is_callable(tx)
            ):
                seq = args[0].unpack_var_sequence(tx)
                func = args[1].call_function

            else:
                raise unimplemented("Unsupported arguments for itertools.accumulate")

            items = []
            acc = None
            for item in seq:
                if acc is None:
                    acc = item
                else:
                    try:
                        acc = func(tx, item, acc)
                    except Exception:
                        raise unimplemented(
                            f"Unexpected failure in invoking function during accumulate. Failed running func {func}({item}{acc})"
                        )
                items.append(acc)

            return variables.ListIteratorVariable(
                items, mutable_local=MutableLocal(), **options
            )
        elif (
            self.value is itertools.combinations
            and not kwargs
            and len(args) == 2
            and args[0].has_unpack_var_sequence(tx)
            and args[1].is_python_constant()
        ):
            iterable = args[0].unpack_var_sequence(tx)
            r = args[1].as_python_constant()

            items = []
            for item in itertools.combinations(iterable, r):
                items.append(variables.TupleVariable(list(item), **options))
            return variables.ListIteratorVariable(
                items, mutable_local=MutableLocal(), **options
            )
        elif (
            self.value is functools.wraps
            and not kwargs
            and len(args) == 1
            and args[0].source
        ):

            def wraps(fn):
                if isinstance(fn, variables.NestedUserFunctionVariable):
                    return fn.clone(wraps_source=args[0].source)
                unimplemented(f"functools.wraps({fn})")

            return variables.LambdaVariable(wraps, **options)
        elif self.value is collections.deque and not kwargs:
            if len(args) == 0:
                items = []
            elif len(args) == 1 and args[0].has_unpack_var_sequence(tx):
                items = args[0].unpack_var_sequence(tx)
            else:
                unimplemented("deque() with more than 1 arg not supported")
            return variables.lists.DequeVariable(
                items, mutable_local=MutableLocal(), **options
            )
        elif self.value is functools.partial:
            if not args:
                unimplemented("functools.partial malformed")
            # The first arg, a callable (the ctor below will assert on types)
            fn = args[0]
            rest_args = args[1:]
            # guards for the produced FunctoolsPartialVariable are installed in FunctoolsPartialVariable ctor from the
            # args and keywords
            return variables.functions.FunctoolsPartialVariable(
                fn, args=rest_args, keywords=kwargs, **options
            )
        else:
            try:
                path = inspect.getfile(self.value)
            except TypeError:
                path = f"Builtin {self.value.__name__}"
            unimplemented(
                f"call_function {self.value.__qualname__} in skip_files {path}"
            )


class TypingVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__getitem__" and len(args) == 1:
            return variables.ConstantVariable(
                self.value[args[0].as_python_constant()],
                **VariableTracker.propagate(self, args),
            )
        unimplemented("typing")

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value


@functools.lru_cache(maxsize=1)
def get_np_to_tnp_map():
    from ..utils import NP_TO_TNP_MODULE

    np_fn_to_tnp_fn = {}

    for np_mod, tnp_mod in NP_TO_TNP_MODULE.items():
        for fn_name, tnp_fn in tnp_mod.__dict__.items():
            if callable(tnp_fn):
                # some internal details do leak from tnp
                # which are not part of numpy API.
                if np_fn := getattr(np_mod, fn_name, None):
                    np_fn_to_tnp_fn[np_fn] = tnp_fn

    return np_fn_to_tnp_fn


class NumpyVariable(VariableTracker):
    """
    Wrapper around `numpy.*`. Currently, is able to trace a small subset of numpy functions as well as numpy dtypes.
    """

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        if not config.trace_numpy:
            unimplemented(f"numpy.{self.value}()")

        from ..utils import numpy_to_tensor_wrapper

        from .tensor import NumpyNdarrayVariable

        options = VariableTracker.propagate([[self]], [args], [list(kwargs.values())])
        # lookup method name in tnp. Things like np.dtype(float) are not supported yet.
        if self.value.__name__ == "dtype":
            unimplemented(
                f"numpy dtype function is not supported yet. Got type {type(self.value)}."
            )
        else:  # We are dealing with a callable.
            func = get_np_to_tnp_map().get(self.value)
            if func is None:
                unimplemented(
                    f"Can't find numpy function {self.value} in torch._numpy. "
                    " Please file an issue to request support for this function."
                )

            # TODO(larryliu0820): currently assuming all numpy.* functions are returning a ndarray that can be
            #  wrapped by NumpyNdarrayVariable which is wrong!
            proxy = tx.output.create_proxy(
                "call_function",
                numpy_to_tensor_wrapper(func),
                *proxy_args_kwargs(args, kwargs),
            )
            return NumpyNdarrayVariable.create(tx, proxy, **options)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        unimplemented("numpy")

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    def as_proxy(self):
        # this handles numpy dtype attribute such as np.float32. TODO(larryliu0820): we should split NumpyVariable
        #  into NumpyVariable for instances/objects and NumpyVariable for types.
        if config.trace_numpy and isinstance(self.value, type):
            # retrieve attribute str. E.g., "float32" if given np.float32

            attr = self.value.__name__
            # get tnp equivalent
            tnp_dtype = tnp.dtype(attr)
            # returning a string here because we are assuming all `dtype` kwargs for numpy
            # functions can take an equivalent string and the behavior of the function would
            # be the same as taking a numpy dtype.
            return tnp_dtype.name

        return super().as_proxy()


# Used to keep track of NULLs pushed on the stack for Python 3.11 function calls
class NullVariable(VariableTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "NullVariable"

    def reconstruct(self, codegen):
        if sys.version_info < (3, 11):
            unimplemented("cannot reconstruct NullVariable in < Python 3.11")
        return [create_instruction("PUSH_NULL")]


class DeletedVariable(VariableTracker):
    """Marker used to implement delattr()"""
