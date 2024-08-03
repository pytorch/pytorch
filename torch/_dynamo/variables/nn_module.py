# mypy: ignore-errors

import functools
import inspect
import itertools
import types
from contextlib import contextmanager, nullcontext
from typing import Dict, List, TYPE_CHECKING

import torch.nn

from .. import trace_rules, variables
from ..exc import (
    ObservedException,
    unimplemented,
    UnspecializeRestartAnalysis,
    Unsupported,
)
from ..guards import GuardBuilder, install_guard
from ..mutation_guard import GenerationTracker
from ..source import (
    AttrSource,
    ConstDictKeySource,
    FSDPNNModuleSource,
    GetItemSource,
    NNModuleSource,
    UnspecializedBuiltinNNModuleSource,
    UnspecializedNNModuleSource,
)
from ..utils import (
    get_custom_getattr,
    get_fake_value,
    is_lazy_module,
    is_namedtuple,
    is_safe_constant,
    istensor,
    istype,
    nnmodule_has_hooks,
    object_has_getattribute,
    proxy_args_kwargs,
    set_example_value,
)
from .base import MutableLocal, typestr, VariableTracker
from .functions import invoke_and_store_as_constant
from .lazy import LazyVariableTracker
from .lists import SliceVariable
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


def initialize_lazy_module(tx: "InstructionTranslator", mod, args, kwargs):
    """
    Fairly coupled helper used by NNModuleVariable and UnspecializedNNModuleVariable.

    Used to cause lazy module to be initialized (and delete its init hook) before tracing. Especially
    useful now that 'allowed' modules graph-break on hooks, calling this first ensures there is no hook
    by the time we trace __call__ and thus no graph-break for lazy allowed modules.
    """
    if hasattr(mod, "_initialize_hook"):

        def convert_to_fake(x):
            if is_namedtuple(x):
                return type(x)(*(convert_to_fake(elem) for elem in x))
            elif isinstance(x, dict):
                return {k: convert_to_fake(v) for k, v in x.items()}
            elif isinstance(x, (list, tuple, set)):
                return type(x)(convert_to_fake(elem) for elem in x)
            elif isinstance(x, torch.fx.Proxy):
                return get_fake_value(x.node, tx)
            else:
                return x

        proxy_args, proxy_kwargs = proxy_args_kwargs(args, kwargs)
        fake_args = [convert_to_fake(arg) for arg in proxy_args]
        fake_kwargs = {k: convert_to_fake(v) for k, v in proxy_kwargs.items()}
        mod._infer_parameters(mod, fake_args, fake_kwargs)


@contextmanager
def record_nn_module_stack(module_key: str, source, tx, mod: torch.nn.Module):
    fully_qualified_name = source.name()
    try:
        tx.nn_module_stack[module_key] = (fully_qualified_name, mod.__class__)
        yield
    finally:
        del tx.nn_module_stack[module_key]


def guard_to_detect_forward_monkeypatching(source, mod):
    # Users sometimes patch the forward method of a nn module instance to
    # perform optimizations like quantization. Though this is not a good
    # software practice, but python allows this and Dynamo needs to detect
    # this patching.
    #
    # One way to do this is to add an ID_MATCH guard on every function
    # getting inlined (https://github.com/pytorch/pytorch/pull/124975). But
    # this increased guard overhead by around 20%.
    #
    # To keep the guard overhead down, we just guard on the `forward` being
    # not present in the mod __dict__. The common case of patching forward
    # method adds `forward` in the instance __dict__, whereas the unpatched
    # `forward` sits in the type(mod).__dict__
    if source:
        if "forward" in mod.__dict__ and callable(mod.__dict__["forward"]):
            # Monkeypatched forward method, add an ID_MATCH guard on forward function
            fwd = mod.__dict__["forward"]
            forward_source = AttrSource(source, "forward")
            if type(fwd) is types.MethodType:
                forward_source = AttrSource(forward_source, "__func__")
            install_guard(forward_source.make_guard(GuardBuilder.CLOSURE_MATCH))
        else:
            # Common case - check that the forward key is absent in mod __dict__
            install_guard(
                source.make_guard(
                    functools.partial(
                        GuardBuilder.NOT_PRESENT_IN_GENERIC_DICT, attr="forward"
                    )
                )
            )


class NNModuleVariable(VariableTracker):
    _nonvar_fields = {
        "module_type",
        "module_key",
        "module",
        "nn_module_stack_source",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self, module_type: type, module_key: str, module: torch.nn.Module, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.module_type = module_type
        self.module_key = module_key
        self.module = module
        assert self.source
        self.nn_module_stack_source = self.source

    def get_nn_module_stack_source(self):
        return self.nn_module_stack_source or self.source

    def set_nn_module_stack_source(self, source):
        self.nn_module_stack_source = source

    def python_type(self):
        return self.module_type

    def _wrap_submodule(
        self, tx: "InstructionTranslator", source, submod, *key_extra, **options
    ):
        return

    def unpack_var_sequence(self, tx):
        # implement list/iter/tuple/etc calls
        base = tx.output.get_submodule(self.module_key)
        if isinstance(base, torch.nn.ModuleDict):
            result = []
            for name, submod in base.items():
                name_var = variables.ConstantVariable.create(name)
                tx.output.register_attr_or_module(
                    submod,
                    self.module_key,
                    name,
                    source=NNModuleSource(GetItemSource(self.source, name)),
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
                )
            )
        return result

    def call_hasattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        mod = tx.output.get_submodule(self.module_key)
        result = hasattr(mod, name)
        install_guard(
            NNModuleSource(AttrSource(self.source, name)).make_guard(
                GuardBuilder.HASATTR
            )
        )
        return variables.ConstantVariable.create(result)

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
        raise UnspecializeRestartAnalysis

    def has_key_in_generic_dict(self, tx: "InstructionTranslator", key):
        base = tx.output.get_submodule(self.module_key)

        if object_has_getattribute(base):
            unimplemented("NNModuleVariable with custom __getattribute__")

        if tx.output.side_effects.has_pending_mutation_of_attr(self, key):
            mutated_attr = tx.output.side_effects.load_attr(self, key, deleted_ok=True)
            return not isinstance(mutated_attr, variables.DeletedVariable)

        base_dict = object.__getattribute__(base, "__dict__")
        return key in base_dict

    def _custom_getattr_fallback(self, base, tx, name, options):
        """Check for a __getattr__ and handle it specially if it is implemented"""
        if object_has_getattribute(base):
            unimplemented("torch.nn.Module with a custom __getattribute__ defined")

        getattr_fn = get_custom_getattr(base, ignore_nn_module_getattr=True)
        if getattr_fn is None:
            return None

        if not isinstance(getattr_fn, types.FunctionType):
            unimplemented("torch.nn.Module with a non-function custom __getattr__")

        return variables.UserMethodVariable(getattr_fn, self, **options).call_function(
            tx, [variables.ConstantVariable.create(name)], {}
        )

    def var_getattr(self, tx: "InstructionTranslator", name):
        from .builder import VariableBuilder

        if self.source:
            source = AttrSource(self.source, name)
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

        if name == "__dict__":
            return variables.GetAttrVariable(self, name, source=source)

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
                    base=base, tx=tx, name=name, options={"source": source}
                )
                if result is not None:
                    return result
                # if we can't find a __getattr__, just raise the AttributeError
                raise

        if name == "forward":
            guard_to_detect_forward_monkeypatching(self.source, base)

        if name == "__class__" and not object_member:
            return variables.UserDefinedClassVariable(base.__class__, source=source)

        if object_member:
            out = VariableBuilder(tx, NNModuleSource(source))(subobj)

            if isinstance(out, (NNModuleVariable, UnspecializedNNModuleVariable)):
                # nn_module_stack source is BC surface area. Ensure that
                # mod._modules["linear"] is reflected as mod.linear for
                # nn_module_stack.
                out.set_nn_module_stack_source(
                    AttrSource(self.get_nn_module_stack_source(), name)
                )
            return out

        else:
            if istype(subobj, property):
                if self.source:
                    # Read the class attribute to reach the property
                    source = AttrSource(AttrSource(self.source, "__class__"), name)
                    # Get the getter function
                    source = AttrSource(source, "fget")
                return variables.UserFunctionVariable(
                    subobj.fget,
                    source=source,
                ).call_function(tx, [(self)], {})
            elif istype(subobj, classmethod):
                return variables.UserMethodVariable(
                    subobj.__func__,
                    variables.UserDefinedObjectVariable(type(base)),
                    source=source,
                )
            elif istype(subobj, staticmethod):
                return variables.UserFunctionVariable(
                    subobj.__get__(base), source=source
                )
            elif istype(subobj, types.FunctionType):
                return variables.UserMethodVariable(subobj, self, source=source)
            elif is_safe_constant(subobj) or istensor(subobj):
                # Support possibly common cases of class members
                return VariableBuilder(tx, NNModuleSource(source))(subobj)
            else:
                unimplemented(
                    f"class property {name} - {typestr(base)} {typestr(subobj)}"
                )

        return variables.GetAttrVariable(self, name, source=source)

    def call_function(
        self,
        tx,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        mod = tx.output.get_submodule(self.module_key)

        with record_nn_module_stack(
            self.module_key, self.get_nn_module_stack_source(), tx, mod
        ):
            is_lazy = is_lazy_module(mod)
            if (
                isinstance(mod, torch.nn.Sequential)
                and mod.__class__.forward is torch.nn.Sequential.forward
            ):
                if nnmodule_has_hooks(mod):
                    # We do not want to unroll sequential if it has hooks, since evaporating it
                    # will cause hooks to not fire!
                    # This terminates and restart the tracing process
                    self.convert_to_unspecialized(tx)

                # Unroll sequential
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
            #
            # NB: torch.nn.utils.parametrize changes the class type of a
            # parametrized module such that its __module__ points to
            # "torch.nn.utils.parametrize".
            if (
                tx.output.is_root_tracer()
                and mod.__module__.startswith(("torch.nn.", "torch.ao."))
                and mod.__module__ != "torch.nn.utils.parametrize"
            ):
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
                    fn_source = AttrSource(self.source, "forward")
                else:
                    fn = mod._call_impl
                    fn_source = AttrSource(self.source, "_call_impl")
                if istype(fn, types.MethodType):
                    fn = fn.__func__
                    fn_source = AttrSource(fn_source, "__func__")
                    args = [self] + args
                else:
                    assert istype(fn, types.FunctionType)
                return tx.inline_user_function_return(
                    variables.UserFunctionVariable(fn, source=fn_source),
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

        key = self.module_key
        module = tx.output.get_submodule(key)

        def generic_call_method_helper(name):
            # Helper function to put a `call_method` node in FX graph,
            # with nn.Module as the first arg.
            mod_proxy = tx.output.create_proxy(
                "get_attr",
                self.module_key,
                (),
                {},
            )
            set_example_value(mod_proxy.node, module)

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
            with record_nn_module_stack(
                self.module_key, self.get_nn_module_stack_source(), tx, module
            ):
                return generic_call_method_helper(name)

        if name == "_check_input_dim" and trace_rules.is_torch_inline_allowed(
            inspect.getfile(module.__class__._check_input_dim)
        ):
            return ConstantVariable.create(True)

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
            )

        if constant:
            fn = getattr(module, name)
            name = f"{module.__class__.__name__}_{name}_result"
            return invoke_and_store_as_constant(tx, fn, name, args, kwargs)

        def assert_all_args_kwargs_const():
            if not all(
                x.is_python_constant() for x in itertools.chain(args, kwargs.values())
            ):
                unimplemented(f"non-const NNModule method {name}")

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
                    )
                )
            return ListIteratorVariable(result, mutable_local=MutableLocal())

        def named_embed(name, obj):
            return TupleVariable(
                [
                    ConstantVariable.create(name),
                    tx.output.register_attr_or_module(
                        obj,
                        key,
                        name,
                        source=NNModuleSource(gen_source(self.source, name)),
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
            tx.output.guard_on_key_order.add(AttrSource(self.source, "_modules").name())
            assert not (args or kwargs)
            result = []
            for name, submod in module.named_children():
                result.append(named_embed(name, submod))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == "named_parameters":
            tx.output.guard_on_key_order.add(
                AttrSource(self.source, "_parameters").name()
            )
            result = []
            for name, param in module.named_parameters(
                **get_kwargs("prefix", "recurse")
            ):
                result.append(named_embed(name, param))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == "named_buffers":
            tx.output.guard_on_key_order.add(AttrSource(self.source, "_buffers").name())
            result = []
            for name, buffer in module.named_buffers(
                **get_kwargs("prefix", "recurse", "remove_duplicate")
            ):
                result.append(named_embed(name, buffer))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == "named_modules":
            tx.output.guard_on_key_order.add(AttrSource(self.source, "_modules").name())
            result = []
            for name, submod in module.named_modules(
                **get_kwargs("memo", "prefix", "remove_duplicate")
            ):
                result.append(named_embed(name, submod))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == "children":
            tx.output.guard_on_key_order.add(AttrSource(self.source, "_modules").name())
            assert not (args or kwargs)
            return wrap_values(module.named_children())
        elif name == "modules":
            tx.output.guard_on_key_order.add(AttrSource(self.source, "_modules").name())
            return wrap_values(module.named_modules())
        elif name == "parameters":
            tx.output.guard_on_key_order.add(
                AttrSource(self.source, "_parameters").name()
            )
            return wrap_values(module.named_parameters(**get_kwargs("recurse")))
        elif name == "buffers":
            tx.output.guard_on_key_order.add(AttrSource(self.source, "_buffers").name())
            return wrap_values(module.named_buffers(**get_kwargs("recurse")))
        elif name == "keys":
            assert not (args or kwargs)
            result = []
            for name in module.keys():
                result.append(ConstantVariable.create(name))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == "values":
            assert not (args or kwargs)
            return wrap_values(module.items())
        elif name == "items":
            assert not (args or kwargs)
            result = []
            for name, submod in module.items():
                result.append(named_embed(name, submod))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == "__len__":
            assert not (args or kwargs)
            return ConstantVariable.create(len(module))
        elif (
            name == "__contains__"
            and isinstance(module, (torch.nn.ModuleDict, torch.nn.ParameterDict))
            and args
            and args[0].is_python_constant()
        ):
            return ConstantVariable.create(
                args[0].as_python_constant() in module._modules
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
                    variables.UserFunctionVariable(fn, source=src),
                    [self] + list(args),
                    kwargs,
                )

            assert self.source

            if isinstance(args[0], SliceVariable):
                # TODO(anijain2305,export-team) - Remove this if condition when inlining of inbuilt nn modules is
                # enabled for export.
                if tx.output.export:
                    # Build a TupleVariable of NNModules
                    result = []

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
                            )
                        )

                    new_module = module[args[0].as_python_constant()]
                    new_module_variable = tx.output.register_attr_or_module(
                        new_module,
                        f"{self}.__getitem__(slice)",
                        source=NNModuleSource(
                            GetItemSource(self.source, args[0].as_python_constant())
                        ),
                    )
                    return new_module_variable
                else:
                    # slice on nn module results in a creation of new module instance, so we need to make it sourceless.
                    # Convert to unspecialized so that UnspecializedNNModule variable can take care of it.
                    self.convert_to_unspecialized(tx)

            from .tensor import SymNodeVariable

            if isinstance(args[0], SymNodeVariable):
                key = args[0].evaluate_expr(tx.output)
            elif args[0].is_python_constant():
                key = args[0].as_python_constant()
            else:
                unimplemented(f"getitem on NNModuleVariable with key {args[0]}")

            submod = module[key]
            return tx.output.register_attr_or_module(
                submod,
                self.module_key,
                key,
                source=NNModuleSource(GetItemSource(self.source, key)),
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
            fn_source = AttrSource(AttrSource(self.source, name), "__func__")
            return tx.inline_user_function_return(
                variables.UserFunctionVariable(fn, source=fn_source),
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
    _nonvar_fields = {
        "value_type",
        "is_state_mutated",
        "nn_module_stack_source",
        *UserDefinedObjectVariable._nonvar_fields,
    }

    """
    The above class will specialize on the id() of a module and place
    parameters on the torch.fx.GraphModule.  Giving one graph per
    module instance.  This version treats nn.Modules() like other user
    defined objects and will pass parameters into the FX graph as inputs.
    Giving one graph per module class.
    """

    def __init__(self, value, **kwargs) -> None:
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
        self.is_state_mutated = False
        # nn_module_stack_source is used to ensure BC for nn_module_stack.
        # Downstream users prefer mod.linear instead of mod._modules['linear']
        # as the module stack. When Dynamo inlines the __getattr__ method, we
        # cannot use self.source for nn_module_stack because it will be similar
        # to mod._modules['linear']. In these cases, we set the
        # nn_module_stack_source appropriately to resemble mod.linear.
        self.nn_module_stack_source = self.source

    def _wrap_source(self, attr_source):
        if not isinstance(attr_source, UnspecializedNNModuleSource):
            return UnspecializedNNModuleSource(attr_source)
        return attr_source

    def get_nn_module_stack_source(self):
        return self.nn_module_stack_source or self.source

    def set_nn_module_stack_source(self, source):
        self.nn_module_stack_source = source

    @staticmethod
    @functools.lru_cache(None)
    def _nn_module_method_ids():
        # Allow __setattr__ to fall through to base class handler
        supported = {torch.nn.Module.__setattr__, torch.nn.Module.__init__}
        return {
            id(x.__code__)
            for x in torch.nn.Module.__dict__.values()
            if hasattr(x, "__code__") and x not in supported
        }

    def unpack_var_sequence(self, tx):
        try:
            fn = inspect.getattr_static(self.value_type, "__iter__")
        except AttributeError as e:
            raise NotImplementedError from e

        if fn in (
            torch.nn.ModuleList.__iter__,
            torch.nn.ParameterList.__iter__,
            torch.nn.Sequential.__iter__,
        ):
            # The program can mutate the nn module object but the saved `value`
            # will not reflect the mutations. So, trace through the `__iter__`
            # function to reflect any tracked mutations.
            return tx.inline_user_function_return(
                variables.UserFunctionVariable(fn),
                [
                    self,
                ],
                {},
            ).unpack_var_sequence(tx)

        return super().unpack_var_sequence(tx)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        mod = self.value
        # see comment on lazy module handling in NNModuleVariable.call_function for context
        if is_lazy_module(mod):
            if mod.cls_to_become is not None:
                self.value_type = mod.cls_to_become
            initialize_lazy_module(tx, mod, args, kwargs)
        name = "_call_impl"
        fn = getattr(self.value_type, name)

        # Check if we can short circuit nn.Module._call_impl to the forward
        # method.  NB - This is done to reduce the compile time of Dynamo.
        if fn is torch.nn.Module._call_impl and "forward" not in mod.__dict__:
            forward_method = inspect.getattr_static(mod, "forward")
            if isinstance(forward_method, types.FunctionType):
                globals_vt = tx.nn_modules_globals_vt
                if not (
                    self.var_getattr(tx, "_backward_hooks").realize().len()
                    or self.var_getattr(tx, "_backward_pre_hooks").realize().len()
                    or self.var_getattr(tx, "_forward_hooks").realize().len()
                    or self.var_getattr(tx, "_forward_pre_hooks").realize().len()
                    or globals_vt.var_getattr(tx, "_global_backward_pre_hooks").len()
                    or globals_vt.var_getattr(tx, "_global_backward_hooks").len()
                    or globals_vt.var_getattr(tx, "_global_forward_hooks").len()
                    or globals_vt.var_getattr(tx, "_global_forward_pre_hooks").len()
                ):
                    name = "forward"
                    fn = self.value_type.forward

        if self.source:
            source = AttrSource(AttrSource(self.source, "__class__"), name)
        else:
            source = None

        guard_to_detect_forward_monkeypatching(self.source, mod)

        ctx = (
            record_nn_module_stack(
                str(id(mod)), self.get_nn_module_stack_source(), tx, mod
            )
            if self.source
            else nullcontext()
        )
        with ctx:
            return variables.UserFunctionVariable(fn, source=source).call_function(
                tx, [self] + list(args), kwargs
            )

    def trace_supported_methods(
        self, tx: "InstructionTranslator", method, name, args, kwargs
    ):
        def get_kwargs(*names):
            fn = getattr(self.value, name)
            bound_args = inspect.signature(fn).bind(
                *([x.as_python_constant() for x in args]),
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
            bound_args.apply_defaults()
            bound_args = bound_args.arguments
            return {k: bound_args[k] for k in names}

        def get_current_parameters(module_var):
            params_dict = module_var.var_getattr(tx, "_parameters").realize().items
            assert isinstance(params_dict, dict)
            params_list = list(params_dict.values())
            params_list = [param.realize() for param in params_list]
            # Account for mod.param = None
            params_list = [
                param
                for param in params_list
                if isinstance(param, variables.TensorVariable)
            ]
            return params_list

        def collect_parameters(module_var, recurse):
            params_list = []
            assert isinstance(module_var, UnspecializedNNModuleVariable)
            params_list = get_current_parameters(module_var)
            modules_dict = module_var.var_getattr(tx, "_modules").realize()
            if recurse:
                for submodule_var in modules_dict.items.values():
                    assert isinstance(submodule_var, UnspecializedNNModuleVariable)
                    params_list.extend(collect_parameters(submodule_var, recurse))
            return params_list

        if method is torch.nn.Module.parameters:
            if self.source:
                tx.output.guard_on_key_order.add(
                    AttrSource(self.source, "_parameters").name()
                )
            recurse = get_kwargs("recurse")["recurse"]
            params_list = collect_parameters(self, recurse=recurse)

            # Account for duplicated params
            deduplicated_params = list(dict.fromkeys(params_list).keys())

            return variables.ListIteratorVariable(
                deduplicated_params, mutable_local=MutableLocal()
            )
        else:
            raise AssertionError(
                "Discrepancy between is_supported_nn_module_method and trace_supported_methods"
            )

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name in ["_call_impl", "_wrapped_call_impl"]:
            fn = getattr(self.value_type, name)
            if self.source:
                source = AttrSource(AttrSource(self.source, "__class__"), name)
            else:
                source = None

            return variables.UserFunctionVariable(fn, source=source).call_function(
                tx, [self] + list(args), kwargs
            )

        if name not in getattr(self.value, "__dict__", {}):
            try:
                method = inspect.getattr_static(type(self.value), name)
            except AttributeError:
                method = None

            if self.is_supported_nn_module_method(method):
                return self.trace_supported_methods(tx, method, name, args, kwargs)

            if isinstance(method, staticmethod):
                source = AttrSource(
                    AttrSource(AttrSource(self.source, "__class__"), name), "__func__"
                )
                return tx.inline_user_function_return(
                    variables.UserFunctionVariable(method.__func__, source=source),
                    args,
                    kwargs,
                )

            if (
                hasattr(method, "__code__")
                and id(method.__code__) in self._nn_module_method_ids()
            ):
                unimplemented(f"UnspecializedNNModuleVariable missing {name}")

            # "_parameters" in self.value.__dict__ checks that module is initialized
            if name == "__setattr__" and "_parameters" in self.value.__dict__:
                # Record if mutations happens on parameters/buffers/modules. The
                # mutations on these are not tracked by base class
                # UserDefinedObject vt. This will be used later to graph break
                # on seeing a paramters() and family calls.
                # TODO(anijain2305) - This might not be needed if we let Dynamo
                # inline both getattr and setattr. In that case, it should see
                # the lowest level dicts - _parameters and family and
                # automatically track mutations on those. Investigate if that
                # can be done.
                attr_name = args[0].as_python_constant()
                value = args[1]

                # This is reverse engineered by looking at nn module __setattr__
                # logic.
                if (
                    isinstance(value, variables.TensorVariable)
                    and value.python_type() is torch.nn.Parameter
                ) or attr_name in self.value.__dict__["_parameters"]:
                    # Handle parameters
                    self.is_state_mutated = True
                elif attr_name in self.value.__dict__["_buffers"]:
                    # Handle buffers
                    self.is_state_mutated = True
                elif (
                    isinstance(
                        value,
                        (
                            variables.NNModuleVariable,
                            variables.UnspecializedNNModuleVariable,
                        ),
                    )
                    or attr_name in self.value.__dict__["_modules"]
                ):
                    # Handle submodules
                    self.is_state_mutated = True

            if method is torch.nn.Module.__setattr__ and isinstance(
                args[1], variables.DeletedVariable
            ):
                # Trace through __delattr__ to track mutations on the module
                # members like `_modules``.
                return tx.inline_user_function_return(
                    variables.UserFunctionVariable(torch.nn.Module.__delattr__),
                    [self, args[0]],
                    kwargs,
                )

        return super().call_method(tx, name, args, kwargs)

    def getattr_helper(self, tx: "InstructionTranslator", field, name_vt):
        dict_vt = self.var_getattr(tx, field)
        if isinstance(dict_vt, variables.ConstDictVariable):
            return dict_vt.maybe_getitem_const(name_vt)
        return None

    def var_getattr(self, tx: "InstructionTranslator", name):
        # Allow skipping of empty hook dict guards on inbuilt nn modules
        if name in (
            "_backward_hooks",
            "_backward_pre_hooks",
            "_forward_hooks",
            "_forward_pre_hooks",
        ):
            # For empty hooks, make an EMPTY_NN_MODULE_HOOKS_DICT. This allows us to control the installation of empty
            # hooks guard via skip_nnmodule_hook_guards
            if not tx.output.side_effects.has_pending_mutation_of_attr(
                self, name
            ) and self.value.__module__.startswith(("torch.nn.", "torch.ao.")):
                hooks_dict = getattr(self.value, name)
                if isinstance(hooks_dict, dict) and len(hooks_dict) == 0:
                    if self.source:
                        hooks_source = AttrSource(self.source, name)
                        install_guard(
                            hooks_source.make_guard(
                                GuardBuilder.EMPTY_NN_MODULE_HOOKS_DICT
                            )
                        )
                    return variables.ConstDictVariable({})

        # For non-empty hook dicts, one way is to just fallback to VariableBuilder and create a ConstDictVariable.
        # However, ConstDictVariable guards on keys. This can cause recompiles when the same hook is installed for
        # differnt nn module instances, because the key keeps changing (look more into RemovableHandle to understand why
        # key changes - also related https://github.com/pytorch/pytorch/issues/125836). Here, we carefully craft a
        # ConstDictVariable to avoid any guard on the keys.
        if (
            self.source
            and name
            in (
                "_forward_pre_hooks",
                "_forward_hooks",
            )
            and not tx.output.side_effects.has_pending_mutation_of_attr(self, name)
        ):
            hooks_dict = getattr(self.value, name)
            hooks_dict_source = AttrSource(self.source, name)
            install_guard(hooks_dict_source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            tx.output.guard_on_key_order.add(hooks_dict_source.name())

            def build_key_value(i, k, v):
                # Make key sourceless to avoid any guard on it
                key = variables.ConstantVariable.create(k)

                # Instead of using dict[key] to access the value, use a dict[dict.keys()[index]] to access the
                # value. This removes the reliance on the actual key value.
                source_key = ConstDictKeySource(hooks_dict_source, i)
                source_value = GetItemSource(hooks_dict_source, source_key)
                value = LazyVariableTracker.create(v, source_value)
                return key, value

            result = dict(
                build_key_value(i, k, v) for i, (k, v) in enumerate(hooks_dict.items())
            )

            return variables.ConstDictVariable(
                result, type(hooks_dict), source=hooks_dict_source
            )
        return super().var_getattr(tx, name)

    def manually_trace_nn_module_getattr(self, tx: "InstructionTranslator", name):
        """
        Dynamo tracing of nn.Module __getattr__ can be expensive if the model
        has deep submodule hierarchy. Since the __getattr__ is stable, we can
        directly look into the underlying datastructures. This saves a lot of
        compilation time.
        """
        name_vt = variables.ConstantVariable(name)
        out = self.getattr_helper(tx, "_parameters", name_vt)
        if out is None:
            out = self.getattr_helper(tx, "_modules", name_vt)
        if out is None:
            out = self.getattr_helper(tx, "_buffers", name_vt)
        if out is None:
            raise ObservedException(f"object has no attribute {name}")
        return out


class UnspecializedBuiltinNNModuleVariable(UnspecializedNNModuleVariable):
    """
    Differentiates between builtin nn modules (e.g. torch.nn.Linear) and user defined nn modules.
    """

    def _wrap_source(self, attr_source):
        if not isinstance(attr_source, UnspecializedBuiltinNNModuleSource):
            return UnspecializedBuiltinNNModuleSource(attr_source)
        return attr_source


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

    def __init__(self, value, **kwargs) -> None:
        source = kwargs.get("source", None)
        assert (
            source is not None
        ), "FSDPManagedNNModule depends on having an accurate source to control guarding."

        super().__init__(value=value, **kwargs)
        self.source = source

    def _wrap_source(self, attr_source):
        if not isinstance(
            attr_source, (FSDPNNModuleSource, UnspecializedNNModuleSource)
        ):
            if torch._dynamo.config.skip_fsdp_guards:
                return FSDPNNModuleSource(attr_source)
            else:
                return UnspecializedNNModuleSource(attr_source)
        return attr_source
