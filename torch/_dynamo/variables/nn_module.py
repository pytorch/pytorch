import functools
import inspect
import itertools
import re
import types
from contextlib import contextmanager
from typing import Dict, List

import torch.nn

from .. import skipfiles, variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..source import AttrSource, GetItemSource, NNModuleSource, NotNNModuleSource
from ..utils import (
    is_lazy_module,
    is_safe_constant,
    istensor,
    istype,
    proxy_args_kwargs,
)
from .base import MutableLocal, typestr, VariableTracker
from .functions import invoke_and_store_as_constant
from .lists import SliceVariable
from .user_defined import UserDefinedObjectVariable


class NNModuleVariable(VariableTracker):
    _nonvar_fields = ["module_type", "module_key"]

    def __init__(self, module_type: type, module_key: str, **kwargs):
        super(NNModuleVariable, self).__init__(**kwargs)
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
            subobj = inspect.getattr_static(base, name)
            object_member = False

        if name == "__class__" and not object_member:
            return variables.UserDefinedClassVariable(base.__class__, **options)

        if object_member:
            return VariableBuilder(tx, NNModuleSource(source))(subobj)
        else:
            if istype(subobj, property):
                return variables.UserFunctionVariable(
                    subobj.fget, guards=guards
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

    def call_function(
        self,
        tx,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        mod = tx.output.get_submodule(self.module_key)

        @contextmanager
        def record_nn_module_stack():
            try:
                tx.nn_module_stack[self.module_key] = type(mod)
                yield
            finally:
                del tx.nn_module_stack[self.module_key]

        with record_nn_module_stack():
            is_lazy = is_lazy_module(mod)
            if (
                isinstance(mod, torch.nn.Sequential)
                and mod.__class__.forward is torch.nn.Sequential.forward
            ):
                # unroll Sequential()
                assert not kwargs
                (arg,) = args
                for idx, submod in enumerate(mod):
                    tx.call_function(
                        tx.output.register_attr_or_module(
                            submod,
                            self.module_key,
                            idx,
                            source=NNModuleSource(GetItemSource(self.source, idx)),
                            **options,
                        ),
                        [arg],
                        {},
                    )
                    arg = tx.pop()
                return arg
            elif is_allowed(mod.__class__):
                # The module type will change after it is called
                if is_lazy:
                    self.module_type = mod.cls_to_become

                return variables.TensorVariable.create(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_module",
                        self.module_key,
                        *proxy_args_kwargs(args, kwargs),
                        current_tx=tx,
                    ),
                    **options,
                )

            else:
                # for lazy modules, run the pre-hooks which will update the type
                # TODO mlazos: we don't fully support all of the hooks that exist,
                # so restrict using __call__ only to lazy modules for now
                if is_lazy:
                    fn = mod.__class__.__call__
                else:
                    fn = mod.__class__.forward

                return tx.inline_user_function_return(
                    variables.UserFunctionVariable(fn, **options),
                    [self] + args,
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

        if name == "forward":
            return self.call_function(tx, args, kwargs)

        if name == "_check_input_dim" and skipfiles.is_torch_inline_allowed(
            inspect.getfile(module.__class__._check_input_dim)
        ):
            return ConstantVariable(True, **options)

        if name == "_get_item_by_idx":
            assert args[1].is_python_constant()
            assert isinstance(args[0], TupleVariable)
            mod_var = args[0].items[args[1].value]
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

        def wrap_values(items, getsource=AttrSource):
            result = []
            for name, submod in items:
                # layer.0.foo => layer[0].foo
                name = re.sub(r"[.]([0-9]+)([.]|$)", r"[\1]\2", name)
                src = NNModuleSource(getsource(self.source, name))
                result.append(
                    tx.output.register_attr_or_module(
                        submod,
                        key,
                        name,
                        source=src,
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
                        source=NNModuleSource(GetItemSource(self.source, name)),
                        **options,
                    ),
                ]
            )

        if name == "children":
            assert not (args or kwargs)
            return wrap_values(module.named_children())
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
        elif name == "parameters":
            return wrap_values(module.named_parameters(**get_kwargs("recurse")))
        elif name == "values":
            assert not (args or kwargs)
            return wrap_values(module.items(), GetItemSource)
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
            assert type(module).__getitem__ in (
                torch.nn.ModuleDict.__getitem__,
                torch.nn.ModuleList.__getitem__,
                torch.nn.ParameterList.__getitem__,
                torch.nn.Sequential.__getitem__,
            ), typestr(module)
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
        elif name == "_get_abs_string_index":
            # Inline the function
            fn = getattr(module, name).__func__
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
            # TODO(voz): Refactor this into a generic as_proxy() for nn module
            # We use variations of this pattern in a few places now.
            def make_attr(name):
                node = tx.output.create_proxy(
                    "get_attr",
                    name,
                    tuple(),
                    {},
                )
                return node

            # Bind in self
            tx.output.register_attr_or_module(
                module,
                self.module_key,
                self.module_key,
                source=NNModuleSource(GetItemSource(self.source, self.module_key)),
                **options,
            )
            proxy_for_mod = make_attr(self.module_key)
            proxy_for_mod.node.meta["example_value"] = module

            proxy_args, proxy_kwargs = proxy_args_kwargs(args, kwargs)

            return variables.TensorVariable.create(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method",
                    name,
                    args=(proxy_for_mod, *proxy_args),
                    kwargs=proxy_kwargs,
                    current_tx=tx,
                ),
                **options,
            )
        else:
            return super().call_method(tx, name, args, kwargs)


class UnspecializedNNModuleVariable(UserDefinedObjectVariable):
    """
    The above class will specialize on the id() of a module and place
    parameters on the torch.fx.GraphModule.  Giving one graph per
    module instance.  This version treats nn.Modules() like other user
    defined objects and will pass parameters into the FX graph as inputs.
    Giving one graph per module class.
    """

    def __init__(self, value, **kwargs):
        super(UnspecializedNNModuleVariable, self).__init__(value=value, **kwargs)
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
        except AttributeError:
            raise NotImplementedError()

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

        # TODO mlazos: only support __call__ for lazy modules
        # until we can support a larger swath of python
        if is_lazy_module(self.value):
            fn = self.value_type.__call__
        else:
            fn = self.value_type.forward

        return variables.UserFunctionVariable(fn, **options).call_function(
            tx, [self] + list(args), kwargs
        )

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .builder import VariableBuilder

        options = VariableTracker.propagate(self, args, kwargs.values())
        module = self.value
        if name not in getattr(self.value, "__dict__", {}):
            try:
                method = inspect.getattr_static(type(self.value), name)
            except AttributeError:
                method = None

            if method is torch.nn.Module.parameters:
                options["guards"].add(
                    self.source.make_guard(GuardBuilder.NN_MODULE_PARAM_NAMES)
                )
                items = []
                named_parameters_kwargs = {}
                if len(args) > 0:
                    assert (
                        len(args) == 1
                    ), f"Expected one positional arg for parameters(recurse), got {args}"
                    assert (
                        not kwargs
                    ), f"Expected at most one arg for parameters, got {args} and {kwargs}"
                    named_parameters_kwargs = {"recurse": args[0]}
                elif len(kwargs):
                    assert (
                        len(kwargs) == 1 and "recurse" in kwargs
                    ), f"Expected only 'recurse' kwarg, got {kwargs}"
                    named_parameters_kwargs.update(kwargs)
                for name, value in self.value.named_parameters(
                    **named_parameters_kwargs
                ):
                    items.append(
                        VariableBuilder(tx, AttrSource(self.source, name))(
                            value
                        ).add_options(options)
                    )
                return variables.ListIteratorVariable(
                    items, mutable_local=MutableLocal(), **options
                )
            elif name == "__getitem__":
                assert not kwargs and len(args) == 1
                assert type(module).__getitem__ in (
                    torch.nn.ModuleDict.__getitem__,
                    torch.nn.ModuleList.__getitem__,
                    torch.nn.ParameterList.__getitem__,
                    torch.nn.Sequential.__getitem__,
                ), typestr(module)
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
            # TODO(whc) i think we are catching some functions here, should we be filtering better above?
            elif isinstance(method, classmethod):
                return tx.inline_user_function_return(
                    variables.UserFunctionVariable(method.__func__, **options),
                    [self] + args,
                    kwargs,
                )
            elif isinstance(method, staticmethod):
                return tx.inline_user_function_return(
                    variables.UserFunctionVariable(method.__func__, **options),
                    args,
                    kwargs,
                )
            # TODO(whc) do we still want to filter out some methods? what is this for?
            elif id(method.__code__) in self._nn_module_method_ids():
                unimplemented(f"UnspecializedNNModuleVariable missing {name}")

        return super().call_method(tx, name, args, kwargs)
