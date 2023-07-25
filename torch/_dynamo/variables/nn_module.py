import functools
import inspect
import itertools
import types
from contextlib import contextmanager
from typing import Dict, List

import torch.nn

from .. import skipfiles, variables
from ..allowed_functions import is_allowed
from ..exc import RestartAnalysis, unimplemented, Unsupported
from ..guards import GuardBuilder
from ..mutation_guard import GenerationTracker
from ..source import (
    AttrSource,
    FSDPNNModuleSource,
    GetItemSource,
    NNModuleSource,
    NotNNModuleSource,
)
from ..utils import (
    get_custom_getattr,
    get_fake_value,
    is_lazy_module,
    is_safe_constant,
    istensor,
    istype,
    nnmodule_has_hooks,
    object_has_getattribute,
    proxy_args_kwargs,
)
from .base import MutableLocal, typestr, VariableTracker
from .functions import invoke_and_store_as_constant
from .lists import SliceVariable
from .user_defined import UserDefinedObjectVariable


def initialize_lazy_module(tx, mod, args, kwargs):
    """
    Fairly coupled helper used by NNModuleVariable and UnspecializedNNModuleVariable.

    Used to cause lazy module to be initialized (and delete its init hook) before tracing. Especially
    useful now that 'allowed' modules graph-break on hooks, calling this first ensures there is no hook
    by the time we trace __call__ and thus no graph-break for lazy allowed modules.
    """
    assert len(kwargs) == 0

    if hasattr(mod, "_initialize_hook"):

        def convert_to_fake(x):
            if isinstance(x, torch.fx.Proxy):
                return get_fake_value(x.node, tx)
            else:
                return x

        input = [
            type(arg)([convert_to_fake(x) for x in arg])
            if isinstance(arg, (list, tuple))
            else convert_to_fake(arg)
            for arg in proxy_args_kwargs(args, {})[0]
        ]
        mod._infer_parameters(mod, input)


class NNModuleVariable(VariableTracker):
    _nonvar_fields = ["module_type", "module_key"]

    def __init__(self, module_type: type, module_key: str, **kwargs):
        super().__init__(**kwargs)
        self.module_type = module_type
        self.module_key = module_key
        assert self.source

    def python_type(self):
        return self.module_type

    def _wrap_submodule(self, tx, source, submod, *key_extra, **options):
        return

    def unpack_var_sequence(self, tx):
        # implement list/iter/tuple/etc calls
        base = tx.output.get_submodule(self.module_key)
        options = VariableTracker.propagate([self])
        if isinstance(base, torch.nn.ModuleDict):
            result = []
            for name, submod in base.items():
                name_var = variables.ConstantVariable(name)
                tx.output.register_attr_or_module(
                    submod,
                    self.module_key,
                    name,
                    source=NNModuleSource(GetItemSource(self.source, name)),
                    **options,
                )
                result.append(name_var)
            return result

        assert isinstance(
            base, (torch.nn.ModuleList, torch.nn.ParameterList, torch.nn.Sequential)
        ), typestr(base)
        assert self.source
        result = []
        for idx, submod in enumerate(base):
            result.append(
                tx.output.register_attr_or_module(
                    submod,
                    self.module_key,
                    idx,
                    source=NNModuleSource(GetItemSource(self.source, idx)),
                    **options,
                )
            )
        return result

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        options = VariableTracker.propagate(self)
        mod = tx.output.get_submodule(self.module_key)
        result = hasattr(mod, name)
        return variables.ConstantVariable(result, **options).add_guard(
            NNModuleSource(AttrSource(self.source, name)).make_guard(
                GuardBuilder.HASATTR
            )
        )

    def is_training(self, tx):
        mod = tx.output.get_submodule(self.module_key)
        return getattr(mod, "training", False)

    def convert_to_unspecialized(self, tx):
        """Restart analysis treating this module as an UnspecializedNNModuleVariable"""
        mod = tx.output.get_submodule(self.module_key)
        GenerationTracker.tag(mod)

        # Mark the class dynamic unless its module initialization
        if tx.f_code.co_name != "__init__":
            GenerationTracker.mark_class_dynamic(type(mod))
        raise RestartAnalysis()

    def _custom_getattr_fallback(self, base, tx, name, options):
        """Check for a __getattr__ and handle it specially if it is implemented"""
        if object_has_getattribute(base):
            unimplemented("torch.nn.Module with a custom __getattribute__ defined")

        getattr_fn = get_custom_getattr(base)
        if getattr_fn is None:
            return None

        if not isinstance(getattr_fn, types.FunctionType):
            unimplemented("torch.nn.Module with a non-function custom __getattr__")

        return variables.UserMethodVariable(getattr_fn, self, **options).call_function(
            tx, [variables.ConstantVariable(name)], {}
        )

    def var_getattr(self, tx, name):
        from .builder import VariableBuilder

        options = VariableTracker.propagate(self)
        guards = options.get("guards", set())

        if self.source:
            source = AttrSource(self.source, name)
            options["source"] = source
        else:
            source = None

        base = tx.output.get_submodule(self.module_key)
        base_dict = object.__getattribute__(base, "__dict__")
        object_member = True
        all_class_attribute_names = set()
        for x in inspect.getmro(base.__class__):
            all_class_attribute_names.update(x.__dict__.keys())

        if not self.source:
            unimplemented("GETATTR with no source")

        if name in base_dict:
            subobj = base_dict[name]
        elif (
            "_modules" in base_dict
            and name in base_dict["_modules"]
            and name not in all_class_attribute_names
        ):
            subobj = base_dict["_modules"][name]
        elif "_parameters" in base_dict and name in base_dict["_parameters"]:
            subobj = base_dict["_parameters"][name]
        elif "_buffers" in base_dict and name in base_dict["_buffers"]:
            subobj = base_dict["_buffers"][name]
        else:
            try:
                subobj = inspect.getattr_static(base, name)
                object_member = False
            except AttributeError:
                # see if we can fallback to __getattr__, which is not checked by getattr_static
                result = self._custom_getattr_fallback(
                    base=base, tx=tx, name=name, options=options
                )
                if result is not None:
                    return result
                # if we can't find a __getattr__, just raise the AttributeError
                raise

        if name == "__class__" and not object_member:
            return variables.UserDefinedClassVariable(base.__class__, **options)

        if object_member:
            return VariableBuilder(tx, NNModuleSource(source))(subobj)
        else:
            if istype(subobj, property):
                return variables.UserFunctionVariable(
                    subobj.fget,
                    guards=guards,
                    source=source,
                ).call_function(tx, [(self)], {})
            elif istype(subobj, classmethod):
                return variables.UserMethodVariable(
                    subobj.__func__,
                    variables.UserDefinedObjectVariable(type(base), guards=guards),
                    **options,
                )
            elif istype(subobj, staticmethod):
                return variables.UserFunctionVariable(subobj.__get__(base), **options)
            elif istype(subobj, types.FunctionType):
                return variables.UserMethodVariable(subobj, self, **options)
            elif is_safe_constant(subobj) or istensor(subobj):
                # Support possibly common cases of class members
                return VariableBuilder(tx, NNModuleSource(source))(subobj)
            else:
                unimplemented(f"class property {typestr(base)} {typestr(subobj)}")

        return variables.GetAttrVariable(self, name, **options)

    @contextmanager
    def record_nn_module_stack(self, tx, mod):
        fully_qualified_name = self.source.name()
        try:
            tx.nn_module_stack[self.module_key] = (fully_qualified_name, type(mod))
            yield
        finally:
            del tx.nn_module_stack[self.module_key]

    def call_function(
        self,
        tx,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        mod = tx.output.get_submodule(self.module_key)

        with self.record_nn_module_stack(tx, mod):
            is_lazy = is_lazy_module(mod)
            if (
                isinstance(mod, torch.nn.Sequential)
                and mod.__class__.forward is torch.nn.Sequential.forward
            ):
                # unroll Sequential()
                assert (
                    not is_lazy
                ), "Expected lazy sequential isn't a valid combination?"
                assert not kwargs
                (arg,) = args
                # TODO: Use named_children when it supports remove_duplicate=False.
                for child_name, submod in mod._modules.items():
                    tx.call_function(
                        tx.output.register_attr_or_module(
                            submod,
                            self.module_key,
                            child_name,
                            source=NNModuleSource(AttrSource(self.source, child_name)),
                            **options,
                        ),
                        [arg],
                        {},
                    )
                    arg = tx.pop()
                return arg

            if is_lazy:
                # The module type will change after it is called
                if mod.cls_to_become is not None:
                    self.module_type = mod.cls_to_become

                # The pre-hook runs to initialize the module shapes, then deletes itself.  After this,
                # the module is more or less not lazy and can be treated as a normal module regardless of
                # is_allowed or other variations.
                initialize_lazy_module(tx, mod, args, kwargs)

            # If we are tracing the higher order op, we want Dynamo to step
            # inside the module call so that Dynamo can see the underlying
            # parameters and buffers and raise them as inputs to the graph.
            if tx.output.is_root_tracer() and is_allowed(mod.__class__):
                if nnmodule_has_hooks(
                    mod, check_forward_hooks=True, check_backward_hooks=True
                ):
                    # End of fn, this bubbles up and restarts tracing.
                    self.convert_to_unspecialized(tx)

                from .builder import wrap_fx_proxy

                return wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_module",
                        self.module_key,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                    **options,
                )
            else:
                assert self.source, (
                    "Must provide a valid source in order to inline, "
                    "since inlined function may have default args which must be guarded."
                )
                if isinstance(mod, torch.fx.GraphModule):
                    # TODO: do we want to support __call__ for GM's?
                    # If so at least some changes are needed, we don't allow inlining
                    # the call_wrapped currently, and maybe other issues too
                    fn = mod.forward
                else:
                    fn = mod._call_impl
                fn_source = AttrSource(self.source, "__call__")
                if istype(fn, types.MethodType):
                    fn = fn.__func__
                    fn_source = AttrSource(fn_source, "__func__")
                    args = [self] + args
                else:
                    assert istype(fn, types.FunctionType)
                options["source"] = fn_source
                return tx.inline_user_function_return(
                    variables.UserFunctionVariable(fn, **options),
                    args,
                    kwargs,
                )

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
        constant=False,
    ) -> "VariableTracker":
        from . import ConstantVariable, ListIteratorVariable, TupleVariable

        options = VariableTracker.propagate(self, args, kwargs.values())
        key = self.module_key
        module = tx.output.get_submodule(key)

        def generic_call_method_helper(name):
            # Helper function to put a `call_method` node in FX graph,
            # with nn.Module as the first arg.
            mod_proxy = tx.output.create_proxy(
                "get_attr",
                self.module_key,
                tuple(),
                {},
            )
            mod_proxy.node.meta["example_value"] = module

            proxy_args, proxy_kwargs = proxy_args_kwargs(args, kwargs)

            from .builder import wrap_fx_proxy

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method",
                    name,
                    args=(mod_proxy, *proxy_args),
                    kwargs=proxy_kwargs,
                ),
                **options,
            )

        if name in ["_call_impl", "_wrapped_call_impl"]:
            # Example: `self.layer.__call__(x)`
            # This is used for explicit calling `__call__` in a forward function.
            # Dynamo inlines `__call__`, includes hooks.
            return self.call_function(tx, args, kwargs)
        elif name == "forward":
            # Example: `self.layer.forward(x)`
            # This is used for explicit calling `forward` in a forward function.
            # Dynamo puts `call_method` node in FX, doesn't trigger hooks.
            with self.record_nn_module_stack(tx, module):
                return generic_call_method_helper(name)

        if name == "_check_input_dim" and skipfiles.is_torch_inline_allowed(
            inspect.getfile(module.__class__._check_input_dim)
        ):
            return ConstantVariable(True, **options)

        if name == "_get_item_by_idx":
            assert args[1].is_python_constant()
            assert isinstance(args[0], TupleVariable)
            mod_var = args[0].items[args[1].value]
            if isinstance(mod_var, UnspecializedNNModuleVariable):
                return mod_var
            key = mod_var.module_key
            submod = tx.output.get_submodule(key)
            return tx.output.register_attr_or_module(
                submod,
                key,
                key,
                source=NNModuleSource(GetItemSource(self.source, key)),
                **options,
            )

        if constant:
            fn = getattr(module, name)
            name = f"{module.__class__.__name__}_{name}_result"
            return invoke_and_store_as_constant(tx, fn, name, options, args, kwargs)

        def assert_all_args_kwargs_const():
            if not all(
                x.is_python_constant() for x in itertools.chain(args, kwargs.values())
            ):
                raise unimplemented(f"non-const NNModule method {name}")

        def get_kwargs(*names):
            assert_all_args_kwargs_const()
            fn = getattr(module, name)
            bound_args = inspect.signature(fn).bind(
                *([x.as_python_constant() for x in args]),
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
            bound_args.apply_defaults()
            bound_args = bound_args.arguments
            return {k: bound_args[k] for k in names}

        def wrap_values(items):
            result = []
            for name, submod in items:
                result.append(
                    tx.output.register_attr_or_module(
                        submod,
                        key,
                        name,
                        source=NNModuleSource(gen_source(self.source, name)),
                        **options,
                    )
                )
            return ListIteratorVariable(result, mutable_local=MutableLocal(), **options)

        def named_embed(name, obj):
            return TupleVariable(
                [
                    ConstantVariable(name, **options),
                    tx.output.register_attr_or_module(
                        obj,
                        key,
                        name,
                        source=NNModuleSource(gen_source(self.source, name)),
                        **options,
                    ),
                ]
            )

        def gen_source(source, name):
            name_split = name.split(".")
            if name_split[0] == "":
                return source
            while len(name_split) > 0:
                x = name_split.pop(0)
                source = AttrSource(source, x)
            return source

        if name == "named_children":
            assert not (args or kwargs)
            result = []
            for name, submod in module.named_children():
                result.append(named_embed(name, submod))
            return ListIteratorVariable(result, mutable_local=MutableLocal(), **options)
        elif name == "named_parameters":
            result = []
            for name, param in module.named_parameters(
                **get_kwargs("prefix", "recurse")
            ):
                result.append(named_embed(name, param))
            return ListIteratorVariable(result, mutable_local=MutableLocal(), **options)
        elif name == "named_buffers":
            result = []
            for name, buffer in module.named_buffers(
                **get_kwargs("prefix", "recurse", "remove_duplicate")
            ):
                result.append(named_embed(name, buffer))
            return ListIteratorVariable(result, mutable_local=MutableLocal(), **options)
        elif name == "named_modules":
            result = []
            for name, submod in module.named_modules(
                **get_kwargs("memo", "prefix", "remove_duplicate")
            ):
                result.append(named_embed(name, submod))
            return ListIteratorVariable(result, mutable_local=MutableLocal(), **options)
        elif name == "children":
            assert not (args or kwargs)
            return wrap_values(module.named_children())
        elif name == "modules":
            return wrap_values(module.named_modules())
        elif name == "parameters":
            return wrap_values(module.named_parameters(**get_kwargs("recurse")))
        elif name == "buffers":
            return wrap_values(module.named_buffers(**get_kwargs("recurse")))
        elif name == "keys":
            assert not (args or kwargs)
            result = []
            for name in module.keys():
                result.append(ConstantVariable(name, **options))
            return ListIteratorVariable(result, mutable_local=MutableLocal(), **options)
        elif name == "values":
            assert not (args or kwargs)
            return wrap_values(module.items())
        elif name == "items":
            assert not (args or kwargs)
            result = []
            for name, submod in module.items():
                result.append(named_embed(name, submod))
            return ListIteratorVariable(result, mutable_local=MutableLocal(), **options)
        elif name == "__len__":
            assert not (args or kwargs)
            return ConstantVariable(len(module), **options)
        elif (
            name == "__contains__"
            and isinstance(module, (torch.nn.ModuleDict, torch.nn.ParameterDict))
            and args
            and args[0].is_python_constant()
        ):
            return ConstantVariable(
                args[0].as_python_constant() in module._modules, **options
            )
        elif name == "__getitem__":
            assert not kwargs and len(args) == 1
            builtin_supported = (
                torch.nn.ModuleDict.__getitem__,
                torch.nn.ModuleList.__getitem__,
                torch.nn.ParameterDict.__getitem__,
                torch.nn.ParameterList.__getitem__,
                torch.nn.Sequential.__getitem__,
            )

            if type(module).__getitem__ not in builtin_supported:
                assert isinstance(args[0], variables.ConstantVariable), typestr(args[0])
                key = args[0].as_python_constant()
                assert isinstance(key, (str, int))
                fn = getattr(module, name).__func__

                assert isinstance(fn, types.FunctionType)

                src = AttrSource(AttrSource(self.source, name), "__func__")
                return tx.inline_user_function_return(
                    variables.UserFunctionVariable(fn, source=src, **options),
                    [self] + list(args),
                    kwargs,
                )

            assert self.source

            if isinstance(args[0], SliceVariable):
                # Build a TupleVariable of NNModules
                result = []
                submods = []

                # Turn the slice into the list of integers
                keys = list(range(len(module)))[args[0].as_python_constant()]
                for idx, submod in enumerate(module[args[0].as_python_constant()]):
                    key = keys[idx]
                    src = NNModuleSource(GetItemSource(self.source, key))
                    result.append(
                        tx.output.register_attr_or_module(
                            submod,
                            key,
                            source=src,
                            **options,
                        )
                    )
                    submods.append(submod)

                new_module = torch.nn.Sequential(*submods)
                new_module_variable = tx.output.register_attr_or_module(
                    new_module,
                    f"{self}.__getitem__(slice)",
                    source=NNModuleSource(
                        GetItemSource(self.source, args[0].as_python_constant())
                    ),
                    **options,
                )
                return new_module_variable

            key = args[0].as_python_constant()
            submod = module[key]
            return tx.output.register_attr_or_module(
                submod,
                key,
                args[0].as_python_constant(),
                source=NNModuleSource(GetItemSource(self.source, key)),
                **options,
            )
        elif (
            name == "_get_abs_string_index"
            or (
                isinstance(module, torch.nn.modules.conv._ConvNd)
                and name == "_conv_forward"
            )
            or (
                isinstance(module, torch.nn.modules.conv._ConvTransposeNd)
                and name == "_output_padding"
            )
        ):
            # Inline the function
            fn = getattr(module, name).__func__
            fn_source = AttrSource(self.source, "__func__")
            options["source"] = fn_source
            return tx.inline_user_function_return(
                variables.UserFunctionVariable(fn, **options),
                [self] + args,
                kwargs,
            )
        # A loose heuristic, but seems to be generally good before we drop into the
        # manual handling of inputs
        elif (
            name in module.__class__.__dict__
            and callable(module.__class__.__dict__[name])
            and all(
                isinstance(x, variables.TensorVariable)
                for x in itertools.chain(args, kwargs.values())
            )
        ):
            return generic_call_method_helper(name)
        else:
            return super().call_method(tx, name, args, kwargs)


class UnspecializedNNModuleVariable(UserDefinedObjectVariable):
    _nonvar_fields = ["value_type"]

    """
    The above class will specialize on the id() of a module and place
    parameters on the torch.fx.GraphModule.  Giving one graph per
    module instance.  This version treats nn.Modules() like other user
    defined objects and will pass parameters into the FX graph as inputs.
    Giving one graph per module class.
    """

    def __init__(self, value, **kwargs):
        if type(value) is torch.jit._script.RecursiveScriptModule:
            raise Unsupported(
                "ScriptModules aren't supported in UnspecializedNNModuleVariable"
                " becuase their .forward function isn't a static member of their type"
            )
        if "value_type" in kwargs:
            lazy_value_to_become = getattr(kwargs["value_type"], "cls_to_become", None)
            if type(value) is lazy_value_to_become:
                # We may have cloned a variabletracker for a LazyModule earlier (e.g. tracking side-effects)
                # and then later we called and mutated the LazyModule into a MaterializedModule.
                # We do not do the mutation upon first seeing a LazyModule since we preserve eager semantics to only
                # mutate upon first call, but this requires we update multiple copies of the VariableTracker post-mutation.
                kwargs["value_type"] = type(value)

        super().__init__(value=value, **kwargs)
        if self.source and self.source.is_nn_module():
            # force guard checks even when `not config.guard_nn_modules``
            self.source = NotNNModuleSource(self.source)

    @staticmethod
    @functools.lru_cache(None)
    def _nn_module_method_ids():
        return {
            id(x.__code__)
            for x in torch.nn.Module.__dict__.values()
            if hasattr(x, "__code__")
        }

    def unpack_var_sequence(self, tx):
        from .builder import VariableBuilder

        try:
            fn = inspect.getattr_static(self.value_type, "__iter__")
        except AttributeError as e:
            raise NotImplementedError from e

        if fn in (
            torch.nn.ModuleList.__iter__,
            torch.nn.ParameterList.__iter__,
            torch.nn.Sequential.__iter__,
        ):
            assert self.source
            return [
                VariableBuilder(tx, source=GetItemSource(self.source, idx))(
                    item
                ).add_options(self)
                for idx, item in enumerate(self.value)
            ]

        return super().unpack_var_sequence(tx)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        mod = self.value
        # see comment on lazy module handling in NNModuleVariable.call_function for context
        if is_lazy_module(mod):
            if mod.cls_to_become is not None:
                self.value_type = mod.cls_to_become
            initialize_lazy_module(tx, mod, args, kwargs)
        name = "_call_impl"
        fn = getattr(self.value_type, name)
        if self.source:
            source = AttrSource(AttrSource(self.source, "__class__"), name)
        else:
            source = None

        return variables.UserFunctionVariable(
            fn, source=source, **options
        ).call_function(tx, [self] + list(args), kwargs)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .builder import VariableBuilder

        options = VariableTracker.propagate(self, args, kwargs.values())
        if name in ["_call_impl", "_wrapped_call_impl"]:
            fn = getattr(self.value_type, name)
            if self.source:
                source = AttrSource(AttrSource(self.source, "__class__"), name)
            else:
                source = None

            return variables.UserFunctionVariable(
                fn, source=source, **options
            ).call_function(tx, [self] + list(args), kwargs)

        if name not in getattr(self.value, "__dict__", {}):
            try:
                method = inspect.getattr_static(type(self.value), name)
            except AttributeError:
                method = None

            if method is torch.nn.Module.parameters:
                assert not args or kwargs
                if tx.output.side_effects.has_pending_mutation(self):
                    unimplemented("Module.parameters() with pending mutation")
                options["guards"].add(
                    self.source.make_guard(GuardBuilder.NN_MODULE_PARAM_NAMES)
                )
                items = []
                for name, value in self.value.named_parameters():
                    items.append(
                        VariableBuilder(tx, AttrSource(self.source, name))(
                            value
                        ).add_options(options)
                    )
                return variables.ListIteratorVariable(
                    items, mutable_local=MutableLocal(), **options
                )
            elif isinstance(method, staticmethod):
                source = AttrSource(
                    AttrSource(AttrSource(self.source, "__class__"), name), "__func__"
                )
                return tx.inline_user_function_return(
                    variables.UserFunctionVariable(
                        method.__func__, source=source, **options
                    ),
                    args,
                    kwargs,
                )

            if id(method.__code__) in self._nn_module_method_ids():
                unimplemented(f"UnspecializedNNModuleVariable missing {name}")

        return super().call_method(tx, name, args, kwargs)


class FSDPManagedNNModuleVariable(UnspecializedNNModuleVariable):
    """
    Tracing behavior: trace into submodules and treat them as Unspecialized, do not
    register parameters to the top-level, treat them as function inputs.

    Guards behavior: if 'skip_fsdp_guards', many guards that would be installed
    by a vanilla UnspecializedNNModuleVariable are simply dropped, on the basis
    that a user wrapping their model in FSDP(model) is already opting into a
    requirement to not modify internal model state, which would already break FSDP without
    compilation.
    """

    def __init__(self, value, **kwargs):
        source = kwargs.get("source", None)
        assert (
            source is not None
        ), "FSDPManagedNNModule depends on having an accurate source to control guarding."

        super().__init__(value=value, **kwargs)
        if torch._dynamo.config.skip_fsdp_guards:
            self.source = FSDPNNModuleSource(source)
        else:
            # this makes us behave like a usual UnspecializedNNModuleVariable for guarding purposes
            self.source = NotNNModuleSource(source)
