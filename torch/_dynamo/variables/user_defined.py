import collections
import contextlib
import functools
import importlib
import inspect
import itertools
import random
import sys
import threading
import types
from typing import Dict, List

import torch._dynamo.config

import torch.nn
from torch._guards import TracingContext

from .. import variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, ODictGetItemSource, RandomValueSource
from ..utils import (
    all_hook_names,
    build_checkpoint_variable,
    check_constant_args,
    get_custom_getattr,
    is_namedtuple_cls,
    is_utils_checkpoint,
    istype,
    namedtuple_fields,
    object_has_getattribute,
)
from .base import MutableLocal, VariableTracker
from .ctx_manager import GenericContextWrappingVariable, NullContextVariable
from .dicts import ConstDictVariable


class UserDefinedVariable(VariableTracker):
    pass


class UserDefinedClassVariable(UserDefinedVariable):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def as_python_constant(self):
        return self.value

    def python_type(self):
        return type(self.value)

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        from . import ConstantVariable
        from .builder import VariableBuilder

        source = AttrSource(self.source, name) if self.source is not None else None
        try:
            obj = inspect.getattr_static(self.value, name)
        except AttributeError:
            obj = None
        if isinstance(obj, staticmethod):
            return variables.UserFunctionVariable(
                obj.__get__(self.value), source=source
            )
        elif isinstance(obj, classmethod):
            return variables.UserMethodVariable(obj.__func__, self, source=source)
        elif source and inspect.ismemberdescriptor(obj):
            return VariableBuilder(tx, source)(obj.__get__(self.value))

        if name in getattr(self.value, "__dict__", {}) or ConstantVariable.is_literal(
            obj
        ):
            if source:
                return VariableBuilder(tx, source)(obj)
            elif ConstantVariable.is_literal(obj):
                return ConstantVariable.create(obj)

        return super().var_getattr(tx, name)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if (
            name == "__subclasses__"
            and len(args) == 0
            and not kwargs
            and "__subclasses__" not in self.value.__dict__
        ):
            options = {"mutable_local": MutableLocal()}
            subs_as_vars: List[VariableTracker] = list()
            for sub in self.value.__subclasses__():
                source = AttrSource(tx.import_source(sub.__module__), sub.__name__)
                subs_as_vars.append(
                    variables.UserDefinedClassVariable(sub, source=source)
                )

            return variables.ListVariable(subs_as_vars, **options)

        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from ..side_effects import SideEffects
        from .builder import SourcelessBuilder

        if self.value is contextlib.nullcontext:
            return NullContextVariable()
        elif (
            issubclass(type(self.value), type)
            and hasattr(self.value, "__enter__")
            and hasattr(self.value, "__exit__")
            and check_constant_args(args, kwargs)
            and len(kwargs) == 0  # TODO(ybliang): support kwargs
        ):
            unwrapped_args = [x.as_python_constant() for x in args]
            return GenericContextWrappingVariable(
                unwrapped_args,
                cm_obj=self.value(*unwrapped_args),
            )
        elif is_namedtuple_cls(self.value):
            fields = namedtuple_fields(self.value)
            field_defaults = self.value._field_defaults

            items = list(args)
            items.extend([None] * (len(fields) - len(items)))

            var_tracker_kwargs = {}
            for field_name, var_tracker in zip(fields, items):
                if var_tracker is None:
                    if field_name in kwargs:
                        field_var = kwargs[field_name]
                    else:
                        assert field_name in field_defaults
                        field_var = SourcelessBuilder()(tx, field_defaults[field_name])
                    var_tracker_kwargs[field_name] = field_var

            for name, value in var_tracker_kwargs.items():
                assert name in fields
                items[fields.index(name)] = value

            assert all(x is not None for x in items)
            return variables.NamedTupleVariable(items, self.value)
        elif (
            inspect.getattr_static(self.value, "__new__", None) in (object.__new__,)
            and SideEffects.cls_supports_mutation_side_effects(self.value)
            and self.source
        ):
            var = tx.output.side_effects.track_object_new(
                self.source,
                self.value,
                variables.UnspecializedNNModuleVariable
                if issubclass(self.value, torch.nn.Module)
                else UserDefinedObjectVariable,
                {},
            )
            if (
                inspect.getattr_static(self.value, "__init__", None)
                is torch.nn.Module.__init__
            ):
                tx.output.side_effects.store_attr(
                    var,
                    "__call_nn_module_init",
                    variables.ConstantVariable.create(True),
                )
                return var
            else:
                var.call_method(tx, "__init__", args, kwargs)
                return var
        elif variables.CustomizedDictVariable.is_matching_cls(self.value):
            options = {"mutable_local": MutableLocal()}
            return variables.CustomizedDictVariable.create(
                self.value, args, kwargs, options
            )
        elif variables.DataClassVariable.is_matching_cls(self.value):
            options = {"mutable_local": MutableLocal()}
            return variables.DataClassVariable.create(self.value, args, kwargs, options)

        return super().call_function(tx, args, kwargs)

    def const_getattr(self, tx, name):
        if name == "__name__":
            return self.value.__name__
        return super().const_getattr(tx, name)


class UserDefinedObjectVariable(UserDefinedVariable):
    """
    Mostly objects of defined type.  Catch-all for something where we only know the type.
    """

    _nonvar_fields = {"value", "value_type", *UserDefinedVariable._nonvar_fields}

    def __init__(self, value, value_type=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.value_type = value_type or type(value)
        assert type(value) is self.value_type

    def __str__(self):
        inner = self.value_type.__name__
        if inner in [
            "builtin_function_or_method",
            "getset_descriptor",
            "method_descriptor",
            "method",
        ]:
            inner = str(getattr(self.value, "__name__", None))
        return f"{self.__class__.__name__}({inner})"

    def python_type(self):
        return self.value_type

    @staticmethod
    @functools.lru_cache(None)
    def _supported_random_functions():
        fns = {
            random.random,
            random.randint,
            random.randrange,
            random.uniform,
        }
        return fns

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import (
            BuiltinVariable,
            ConstantVariable,
            TupleVariable,
            UserMethodVariable,
        )


        if (
            self.value.__str__()
            == "<method 'run_backward' of 'torch._C._EngineBase' objects>"
            and name == "__call__"
        ):
            from torch._dynamo import compiled_autograd
            from torch.autograd.variable import Variable
            from .lists import BaseListVariable
            from .tensor import map_fake_tensor_to_tensor_variable

            assert len(args) == 5

            # This is hack, we need to be able to let actual_backward_call be able to
            # access to gm passed to `dummy_fn_to_save_fx`.
            saved_gm = None

            # can't directly wrap gm in a UserFunctionVariable so have to wrap it in a
            # normal python function
            def dummy_user_function_to_inline(gm, input, hook, size):
                # TODO: need to take the output of the gm and assign it back to the input.grad
                return gm(input, hook, size)

            # We need to inline the graphmodule and let dynamo trace through it with inputs.
            # inputs are a bunch of fake_tensors since the compiled_autograd is invoked with
            # a fake tensor(`self.proxy.node.meta.get("example_value")`)
            def actual_backward_call(input, hook, size):
                from .base import MutableLocal
                from .builder import SourcelessBuilder
                from .lists import BaseListVariable

                user_fn_variable = SourcelessBuilder()(
                    tx, dummy_user_function_to_inline
                )
                # gm_variable is a UserDefinedObjectVariable
                gm_variable = SourcelessBuilder()(tx, saved_gm)
                cls = BaseListVariable.cls_for(list)
                # if backward is run without argument, `aten.ones_like` will
                # be used to create a defualt tensor arg. That tensor will not have a
                # TensorVariable assoicatied with it.
                tensor_variables = [
                    map_fake_tensor_to_tensor_variable(x) for x in input
                ]
                input_list_variable = cls(
                    tensor_variables,
                    mutable_local=MutableLocal(),
                )
                # assume hook and size to be empty for now
                hook_variable = cls([], mutable_local=MutableLocal())
                size_variable = cls([], mutable_local=MutableLocal())
                # output of the inline, need to find the corresponding inputs
                res = tx.inline_user_function_return(
                    user_fn_variable,
                    (gm_variable, input_list_variable, hook_variable, size_variable),
                    {},
                )
                # TODO: we should avoid running the actual backward
                return saved_gm(input, hook, size)

            # This is being called in the compiled_autograd's end_capture, it need to return a
            # python function that can execute the backward.
            def dummy_fn_to_save_fx(graph):
                nonlocal saved_gm
                saved_gm = graph
                return actual_backward_call

            # now we want to use compiled_auto_grad to get and inline the fx graph
            tensors_example = tuple(
                [proxy.node.meta.get("example_value") for proxy in args[0].as_proxy()]
            )
            grad_tensors_example = tuple(
                (proxy.node.meta.get("example_value") for proxy in args[1].as_proxy())
            )
            retain_graph = args[2].as_python_constant()
            create_graph = args[3].as_python_constant()
            # ignore the inputs for backward for now
            inputs = ()
            with compiled_autograd.enable(dummy_fn_to_save_fx):
                Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
                    tensors_example,
                    grad_tensors_example,
                    retain_graph,
                    create_graph,
                    inputs,
                    allow_unreachable=True,
                    accumulate_grad=True,
                )
            return ConstantVariable.create(None, **options)

        if name not in getattr(self.value, "__dict__", {}):
            try:
                method = inspect.getattr_static(type(self.value), name)
            except AttributeError:
                method = None
            if method is object.__init__:
                return ConstantVariable.create(None)

            if method is collections.OrderedDict.keys and self.source:
                # subclass of OrderedDict
                assert not (args or kwargs)
                keys = list(self.value.keys())
                assert all(map(ConstantVariable.is_literal, keys))
                install_guard(self.source.make_guard(GuardBuilder.ODICT_KEYS))
                return TupleVariable([ConstantVariable.create(k) for k in keys])

            if (
                method in (collections.OrderedDict.__contains__, dict.__contains__)
                and len(args) == 1
                and isinstance(args[0], (ConstantVariable, BuiltinVariable))
                and inspect.getattr_static(type(self.value), "keys")
                in (collections.OrderedDict.keys, dict.keys)
            ):
                assert not kwargs
                install_guard(self.source.make_guard(GuardBuilder.ODICT_KEYS))
                return ConstantVariable.create(
                    args[0].as_python_constant() in self.value
                )

            if (
                method is collections.OrderedDict.items
                and isinstance(self.value, collections.OrderedDict)
                and self.source
            ):
                assert not (args or kwargs)
                items = []
                keys = self.call_method(tx, "keys", [], {})
                for key in keys.unpack_var_sequence(tx):
                    items.append(
                        TupleVariable(
                            [key, self.odict_getitem(tx, key)],
                        )
                    )
                return TupleVariable(items)

            if method is collections.OrderedDict.__getitem__ and len(args) == 1:
                assert not kwargs
                return self.odict_getitem(tx, args[0])

            # check for methods implemented in C++
            if isinstance(method, types.FunctionType):
                source = (
                    None
                    if self.source is None
                    else AttrSource(AttrSource(self.source, "__class__"), name)
                )
                # TODO(jansel): add a guard to check for monkey patching?
                return UserMethodVariable(method, self, source=source).call_function(
                    tx, args, kwargs
                )

        return super().call_method(tx, name, args, kwargs)

    def is_supported_random(self):
        try:
            return self.value in self._supported_random_functions()
        except TypeError:
            # TypeError: unhashable type
            return False

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .. import trace_rules
        from .builder import VariableBuilder

        if (
            self.is_supported_random()
            and all(k.is_python_constant() for k in args)
            and all(v.is_python_constant() for v in kwargs.values())
        ):
            args = [x.as_python_constant() for x in args]
            kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
            random_call_index = len(tx.random_calls)
            example_value = self.value(*args, **kwargs)
            source = RandomValueSource(random_call_index)
            tx.random_calls.append((self.value, args, kwargs))
            return VariableBuilder(tx, source).wrap_unspecialized_primitive(
                example_value
            )
        elif istype(self.value, types.MethodType):
            func = self.value.__func__
            obj = self.value.__self__
            if (
                func is torch.utils._contextlib._DecoratorContextManager.clone
                and trace_rules.lookup(obj.__class__)
                == variables.TorchCtxManagerClassVariable
                and not (args or kwargs)
            ):
                return variables.TorchCtxManagerClassVariable(
                    obj.__class__
                ).call_function(tx, args, kwargs)

            if (
                func is torch.autograd.grad_mode.inference_mode.clone
                and obj.__class__ is torch.autograd.grad_mode.inference_mode
            ):
                # simulate the inference_mode.clone implementation
                var = variables.ConstantVariable(obj.mode)
                return variables.TorchCtxManagerClassVariable(
                    obj.__class__
                ).call_function(tx, [var], kwargs)
        elif (
            istype(self.value, functools.partial)
            and is_allowed(self.value.func)
            and all(
                variables.ConstantVariable.is_literal(v)
                for v in itertools.chain(self.value.args, self.value.keywords.values())
            )
        ):
            if self.source:
                install_guard(
                    AttrSource(self.source, "func").make_guard(GuardBuilder.ID_MATCH),
                    AttrSource(self.source, "args").make_guard(
                        GuardBuilder.CONSTANT_MATCH
                    ),
                    AttrSource(self.source, "keywords").make_guard(
                        GuardBuilder.CONSTANT_MATCH
                    ),
                )

            partial_args = [
                variables.ConstantVariable.create(v) for v in self.value.args
            ]
            partial_args.extend(args)
            partial_kwargs = {
                k: variables.ConstantVariable.create(v)
                for k, v in self.value.keywords.items()
            }
            partial_kwargs.update(kwargs)
            if is_utils_checkpoint(self.value.func):
                return build_checkpoint_variable().call_function(
                    tx, partial_args, partial_kwargs
                )
            return variables.TorchVariable(self.value.func).call_function(
                tx, partial_args, partial_kwargs
            )
        elif callable(self.value):
            # TODO: UserDefinedObjectVariable(CompiledAutograd) is constructed with SourcelessBuilder
            # so it does not have source
            if self.source:
                install_guard(self.source.make_guard(GuardBuilder.FUNCTION_MATCH))
            return self.call_method(tx, "__call__", args, kwargs)

        return super().call_function(tx, args, kwargs)

    def _check_for_getattribute(self):
        if object_has_getattribute(self.value):
            unimplemented("UserDefinedObjectVariable with custom __getattribute__")

    def _check_for_getattr(self):
        return get_custom_getattr(self.value)

    def _getattr_static(self, name):
        if (
            isinstance(self.value, torch.nn.Module)
            or "__slots__" in self.value.__class__.__dict__
            or type(self.value) == threading.local
        ):
            # getattr_static doesn't work on these
            subobj = getattr(self.value, name)
        else:
            subobj = inspect.getattr_static(self.value, name)
        return subobj

    def var_getattr(self, tx, name):
        from . import ConstantVariable
        from .builder import VariableBuilder

        value = self.value
        source = AttrSource(self.source, name) if self.source else None
        self._check_for_getattribute()
        getattr_fn = self._check_for_getattr()

        class NO_SUCH_SUBOBJ:
            pass

        try:
            subobj = self._getattr_static(name)
        except AttributeError:
            subobj = NO_SUCH_SUBOBJ
            if isinstance(getattr_fn, types.FunctionType):
                return variables.UserMethodVariable(
                    getattr_fn, self, source=source
                ).call_function(tx, [ConstantVariable.create(name)], {})
            elif getattr_fn is not None:
                unimplemented("UserDefined with non-function __getattr__")

        if isinstance(subobj, property):
            return variables.UserMethodVariable(
                subobj.fget, self, source=source
            ).call_function(tx, [], {})
        elif isinstance(subobj, torch.distributions.utils.lazy_property):
            subobj_var = UserDefinedObjectVariable(subobj, source=source)
            return variables.UserMethodVariable(
                subobj.__get__.__func__, subobj_var, source=source
            ).call_function(tx, [self], {})
        elif isinstance(subobj, staticmethod):
            return variables.UserFunctionVariable(
                subobj.__get__(self.value), source=source
            )
        elif isinstance(subobj, classmethod):
            return variables.UserMethodVariable(subobj.__func__, self, source=source)
        elif isinstance(subobj, types.FunctionType) or (
            isinstance(subobj, types.MethodType)
            and isinstance(self.value, torch.nn.Module)
        ):
            # Since we get subobj via self._getattr_static, which may not trigger dynamic lookup.
            # Static lookup can't tell us it's a method or function correctly,
            # so we trigger dynamic lookup here to get the correct type.
            dynamic_subobj = getattr(self.value, name)

            while dynamic_subobj is subobj and hasattr(subobj, "_torchdynamo_inline"):
                subobj = subobj._torchdynamo_inline
                dynamic_subobj = subobj
                source = AttrSource(source, "_torchdynamo_inline") if source else None

            if isinstance(subobj, types.MethodType):
                if dynamic_subobj.__self__ is not self.value:
                    unimplemented("__self__ mismatch for bound method")
                func = subobj.__func__
                source = AttrSource(source, "__func__") if source else None
            else:
                assert isinstance(subobj, types.FunctionType)
                func = subobj

            if inspect.ismethod(dynamic_subobj):
                return variables.UserMethodVariable(func, self, source=source)
            elif inspect.isfunction(dynamic_subobj):
                if is_utils_checkpoint(func):
                    return build_checkpoint_variable(source=source)
                elif is_allowed(func):
                    return variables.TorchVariable(func, source=source)
                return variables.UserFunctionVariable(func, source=source)

        if (
            name in getattr(value, "__dict__", {})
            or ConstantVariable.is_literal(subobj)
            or isinstance(
                subobj,
                (
                    torch.Tensor,
                    torch.nn.Module,
                ),
            )
        ):
            if source:
                return VariableBuilder(tx, source)(subobj)
            elif ConstantVariable.is_literal(subobj):
                return ConstantVariable.create(subobj)

        if (
            name not in getattr(value, "__dict__", {})
            and type(value).__module__.startswith("torch.")
            and "torch.optim" not in type(value).__module__
            and not callable(value)
        ):
            if not source:
                assert getattr(
                    importlib.import_module(type(value).__module__),
                    type(value).__name__,
                ) is type(value)
                source = AttrSource(
                    AttrSource(
                        tx.import_source(type(value).__module__), type(value).__name__
                    ),
                    name,
                )

            return VariableBuilder(tx, source)(subobj)
        options = {"source": source}
        if isinstance(
            subobj,
            (
                torch.distributions.constraints._Interval,
                torch.distributions.constraints._Real,
                torch.distributions.constraints.Constraint,
            ),
        ):
            return UserDefinedObjectVariable(subobj, **options)
        elif isinstance(self.value, torch.nn.Module) and name in all_hook_names:
            assert isinstance(subobj, collections.OrderedDict)
            if not subobj:
                return variables.ConstDictVariable(
                    subobj, collections.OrderedDict, **options
                )

        if name == "__class__":
            return UserDefinedClassVariable(type(self.value), **options)

        return variables.GetAttrVariable(self, name, **options)

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        if tx.output.side_effects.is_attribute_mutation(self):
            try:
                result = tx.output.side_effects.load_attr(self, name, deleted_ok=True)
                return variables.ConstantVariable.create(
                    not isinstance(result, variables.DeletedVariable)
                )
            except KeyError:
                pass
        if self.source:
            install_guard(
                AttrSource(self.source, name).make_guard(GuardBuilder.HASATTR)
            )
        if self._check_for_getattribute() or self._check_for_getattr():
            unimplemented("hasattr with custom __getattr__")

        try:
            self._getattr_static(name)
            return variables.ConstantVariable.create(True)
        except AttributeError:
            return variables.ConstantVariable.create(False)

    def odict_getitem(self, tx, key):
        from .builder import VariableBuilder

        index = (
            key.source
            if ConstDictVariable.is_valid_key(key) and key.source is not None
            else key.as_python_constant()
        )

        return VariableBuilder(
            tx,
            ODictGetItemSource(self.source, index),
        )(collections.OrderedDict.__getitem__(self.value, key.as_python_constant()))


class SourcelessGraphModuleVariable(UserDefinedObjectVariable):
    def __init__(
        self,
        value,
        **kwargs,
    ):
        super().__init__(value, **kwargs)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        fn_variable = variables.UserFunctionVariable(
            self.value.forward.__func__, **options
        )
        args = [self] + args
        return tx.inline_user_function_return(
            fn_variable,
            args,
            kwargs,
        )


class KeyedJaggedTensorVariable(UserDefinedObjectVariable):
    @staticmethod
    def is_matching_object(obj):
        mod = sys.modules.get("torchrec.sparse.jagged_tensor")
        return mod is not None and type(obj) is mod.KeyedJaggedTensor

    def __init__(self, value, **kwargs):
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        assert type(value) is KeyedJaggedTensor
        super().__init__(value, **kwargs)

    def var_getattr(self, tx, name):
        if (
            torch._dynamo.config.force_unspec_int_unbacked_size_like_on_torchrec_kjt
            and self.source is not None
            and name in ("_length_per_key", "_offset_per_key")
        ):
            with TracingContext.patch(force_unspec_int_unbacked_size_like=True):
                return super().var_getattr(tx, name)
        return super().var_getattr(tx, name)


class RemovableHandleVariable(VariableTracker):
    def __init__(
        self,
        mutable_local=None,
        # index of the registration in the side_effects owned register_hook/handle list, used during removal.
        idx=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mutable_local = mutable_local
        self.idx = idx

    def call_method(self, tx, method_name, args, kwargs):
        if method_name == "remove":
            tx.output.side_effects.remove_hook(self.idx)
            return variables.ConstantVariable.create(None)
        super().call_method(tx, method_name, args, kwargs)

    # This reconstruct is actually pretty unique - it does not construct the object from scratch.
    # Handles always come from a register_hook call on a tensor, and so, rerunning that for the codegen of a
    # hook would be incorrect.
    # Instead, the invariant is that codegen has already produced the handle and stored it at a known name.
    def reconstruct(self, codegen):
        if self.user_code_variable_name:
            # It is an invariant that at this point, a STORE_FAST was executed for this name.
            return [codegen.create_load(self.user_code_variable_name)]
        return super().reconstruct(codegen)
