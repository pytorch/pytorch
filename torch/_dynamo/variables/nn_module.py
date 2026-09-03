"""
This module implements variable tracking for PyTorch nn.Module instances during Dynamo tracing.

It provides specialized handling for different types of nn.Module instances through several key classes:

- UnspecializedNNModuleVariable: Provides class-level module tracing, treating nn.Modules like other
  user-defined objects and passing parameters as inputs to the FX graph. This creates one graph per
  module class.

- UnspecializedBuiltinNNModuleVariable: Specifically handles built-in PyTorch modules (e.g. nn.Linear)
  with appropriate optimizations.

- FSDPManagedNNModuleVariable: Special handling for FSDP-wrapped modules with modified guarding behavior
  and parameter handling.

The module integrates with Dynamo's broader tracing functionality to handle module method calls,
parameter access, hooks, and other nn.Module behaviors while maintaining proper scoping and guarding
of module state.
"""

import collections
import functools
import inspect
import re
import types
from contextlib import contextmanager, nullcontext
from typing import Any, TYPE_CHECKING

import torch.nn
from torch._guards import Source

from .. import graph_break_hints, variables
from ..exc import raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..source import (
    AttrSource,
    ConstDictKeySource,
    DictGetItemSource,
    FSDPNNModuleSource,
    GetItemSource,
    UnspecializedNNModuleSource,
)
from ..utils import (
    enumerate_items_with_dict_position,
    get_fake_value,
    is_lazy_module,
    is_namedtuple,
    is_safe_constant,
    istype,
    proxy_args_kwargs,
    unpatched_nn_module_call,
    unpatched_nn_module_call_impl,
)
from .base import VariableTracker
from .lazy import LazyVariableTracker
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


def initialize_lazy_module(
    tx: "InstructionTranslatorBase",
    mod: torch.nn.Module,
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> None:
    """
    Fairly coupled helper used by UnspecializedNNModuleVariable.

    Used to cause lazy module to be initialized (and delete its init hook) before tracing. Especially
    useful now that 'allowed' modules graph-break on hooks, calling this first ensures there is no hook
    by the time we trace __call__ and thus no graph-break for lazy allowed modules.
    """
    if inspect.getattr_static(mod, "_initialize_hook", None) is not None:

        def convert_to_fake(x: Any) -> Any:
            if is_namedtuple(x):
                return type(x)(*(convert_to_fake(elem) for elem in x))
            elif isinstance(x, dict):
                return {k: convert_to_fake(v) for k, v in x.items()}  # type: ignore[misc]
            elif isinstance(x, (list, tuple, set)):
                return type(x)(convert_to_fake(elem) for elem in x)
            elif isinstance(x, torch.fx.Proxy):
                fake = get_fake_value(x.node, tx)
                if isinstance(fake, torch.Tensor) and any(
                    isinstance(s, torch.SymInt) for s in fake.shape
                ):
                    # _infer_parameters runs real ops on the module, so
                    # symbolic shapes must be concretized to their hints.
                    shape = [
                        s.node.hint if isinstance(s, torch.SymInt) else s
                        for s in fake.shape
                    ]
                    if not all(isinstance(s, int) for s in shape):
                        raise AssertionError(
                            f"expected all shape entries to be int, got {shape}"
                        )
                    return torch.empty(  # pyrefly: ignore[no-matching-overload]
                        shape, dtype=fake.dtype, device=fake.device
                    )
                return fake
            else:
                return x

        proxy_args, proxy_kwargs = proxy_args_kwargs(args, kwargs)
        fake_args = [convert_to_fake(arg) for arg in proxy_args]
        fake_kwargs = {k: convert_to_fake(v) for k, v in proxy_kwargs.items()}
        try:
            mod._infer_parameters(mod, fake_args, fake_kwargs)  # type: ignore[operator]
        except AttributeError:
            # Re-raise with the original error message from the AttributeError
            raise_observed_exception(
                AttributeError,
                tx,
                args=["AttributeError during lazy module initialization"],
            )


@contextmanager
def record_nn_module_stack(
    module_key: str,
    source: Source,
    tx: "InstructionTranslatorBase",
    mod: torch.nn.Module,
) -> Any:
    fully_qualified_name = source.name
    # Remove redundant namings
    fully_qualified_name = re.sub(
        r"\._(?:modules|parameters|buffers)\[(['\"])([^'\"\]]+)\1\]",
        r".\2",
        fully_qualified_name,
    )
    num_calls = tx.num_calls.get(fully_qualified_name, 0)
    module_key = f"{module_key}@{num_calls}" if num_calls > 0 else module_key
    try:
        tx.nn_module_stack[module_key] = (fully_qualified_name, mod.__class__)
        tx.num_calls[fully_qualified_name] = num_calls + 1
        yield
    finally:
        del tx.nn_module_stack[module_key]


def guard_to_detect_forward_monkeypatching(
    source: Source | None, mod: torch.nn.Module
) -> None:
    # Users sometimes patch the forward method of a nn module instance to
    # perform optimizations like quantization. Though this is not a good
    # software practice, python allows this and Dynamo needs to detect
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
            # Monkeypatched forward method, guard on call-relevant structure.
            fwd = mod.__dict__["forward"]
            forward_source = AttrSource(source, "forward")
            if type(fwd) is types.MethodType:
                forward_source = AttrSource(forward_source, "__func__")
                install_guard(forward_source.make_guard(GuardBuilder.CLOSURE_MATCH))
            elif isinstance(fwd, functools.partial):
                guard_to_detect_forward_partial_monkeypatching(
                    source, mod, forward_source, fwd
                )
            else:
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


def guard_to_detect_forward_partial_monkeypatching(
    module_source: Source,
    mod: torch.nn.Module,
    partial_source: Source,
    partial_obj: functools.partial[Any],
) -> None:
    install_guard(partial_source.make_guard(GuardBuilder.TYPE_MATCH))

    func_source = AttrSource(partial_source, "func")
    if isinstance(partial_obj.func, functools.partial):
        guard_to_detect_forward_partial_monkeypatching(
            module_source, mod, func_source, partial_obj.func
        )
    else:
        install_guard(func_source.make_guard(GuardBuilder.CLOSURE_MATCH))

    args_source = AttrSource(partial_source, "args")
    install_guard(args_source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
    for i, arg in enumerate(partial_obj.args):
        guard_to_detect_forward_partial_value(
            module_source, mod, GetItemSource(args_source, i), arg
        )

    keywords_source = AttrSource(partial_source, "keywords")
    if partial_obj.keywords is None:
        install_guard(keywords_source.make_guard(GuardBuilder.NONE_MATCH))
        return

    install_guard(keywords_source.make_guard(GuardBuilder.DICT_KEYS_MATCH))
    for key, value in partial_obj.keywords.items():
        guard_to_detect_forward_partial_value(
            module_source, mod, DictGetItemSource(keywords_source, key), value
        )


def guard_to_detect_forward_partial_value(
    module_source: Source,
    mod: torch.nn.Module,
    value_source: Source,
    value: Any,
) -> None:
    if value is mod:
        dupe_guard = make_dupe_guard(value_source, module_source)
        install_guard(value_source.make_guard(dupe_guard or GuardBuilder.ID_MATCH))
    elif isinstance(value, functools.partial):
        guard_to_detect_forward_partial_monkeypatching(
            module_source, mod, value_source, value
        )
    elif type(value) is types.FunctionType:
        install_guard(value_source.make_guard(GuardBuilder.CLOSURE_MATCH))
    elif is_safe_constant(value):
        install_guard(value_source.make_guard(GuardBuilder.CONSTANT_MATCH))
    else:
        install_guard(value_source.make_guard(GuardBuilder.ID_MATCH))


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

    def __init__(self, value: torch.nn.Module, **kwargs: Any) -> None:
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

    def _wrap_source(self, attr_source: Source) -> Source:
        # the vt is already wrapped with UnspecializedNNModuleSource
        return attr_source

    def get_nn_module_stack_source(self) -> Source:
        res = self.nn_module_stack_source or self.source
        if not res:
            raise AssertionError("nn_module_stack_source must not be None")
        return res

    def set_nn_module_stack_source(self, source: Source) -> None:
        self.nn_module_stack_source = source

    @staticmethod
    @functools.cache
    def _nn_module_method_ids() -> set[int]:
        # Allow __setattr__ to fall through to base class handler
        supported = {
            torch.nn.Module.__setattr__,
            torch.nn.Module.__init__,
            torch.nn.Module.__delattr__,
        }
        return {
            id(x.__code__)
            for x in torch.nn.Module.__dict__.values()
            if hasattr(x, "__code__") and x not in supported
        }

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
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
                VariableTracker.build(tx, fn),
                [
                    self,
                ],
                {},
            ).unpack_var_sequence(tx)

        return super().unpack_var_sequence(tx)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        mod = self.value
        if is_lazy_module(mod):  # type: ignore[arg-type]
            if mod.cls_to_become is not None:  # type: ignore[attr-defined]
                self.value_type = mod.cls_to_become  # type: ignore[attr-defined,assignment]
            initialize_lazy_module(tx, mod, args, kwargs)  # type: ignore[arg-type]

        if not isinstance(mod, torch.fx.GraphModule):
            name = "__call__"
            fn = getattr(self.value_type, name)
        else:
            name = "_call_impl"
            fn = getattr(self.value_type, name)

        # Check if we can short circuit nn.Module._call_impl to the forward
        # method.  NB - This is done to reduce the compile time of Dynamo.
        if (
            istype(mod.__call__, types.MethodType)  # type: ignore[operator]
            and istype(mod._call_impl, types.MethodType)  # type: ignore[attr-defined]
            and mod.__call__.__func__ is unpatched_nn_module_call  # type: ignore[operator]
            and mod._call_impl.__func__ is unpatched_nn_module_call_impl  # type: ignore[attr-defined]
            # Consult pending STORE_ATTR side effects too. During tracing the
            # patched forward may not be visible in mod.__dict__ yet.
            and not self.has_key_in_generic_dict(tx, "forward")
        ):
            forward_method = inspect.getattr_static(mod, "forward")
            if isinstance(forward_method, types.FunctionType):
                globals_vt = tx.nn_modules_globals_vt

                def _hooks_dict_len(obj: VariableTracker, attr: str) -> int:
                    vt = obj.tp_getattro_impl(tx, attr)
                    vt = vt.realize() if hasattr(vt, "realize") else vt
                    return vt.len()  # type: ignore[union-attr]

                has_hooks = any(
                    _hooks_dict_len(self, attr)
                    for attr in (
                        "_backward_hooks",
                        "_backward_pre_hooks",
                        "_forward_hooks",
                        "_forward_hooks_with_kwargs",
                        "_forward_pre_hooks",
                        "_forward_pre_hooks_with_kwargs",
                    )
                ) or any(
                    _hooks_dict_len(globals_vt, attr)
                    for attr in (
                        "_global_backward_pre_hooks",
                        "_global_backward_hooks",
                        "_global_forward_hooks",
                        "_global_forward_pre_hooks",
                    )
                )

                if not has_hooks:
                    name = "forward"
                    fn = self.value_type.forward  # type: ignore[attr-defined]

        if self.source:
            source = self.get_source_by_walking_mro(tx, name)
        else:
            source = None

        guard_to_detect_forward_monkeypatching(self.source, mod)  # type: ignore[arg-type]

        ctx = (
            record_nn_module_stack(
                str(id(mod)),
                self.get_nn_module_stack_source(),
                tx,
                mod,  # type: ignore[arg-type]
            )
            if self.source
            else nullcontext()
        )
        with ctx:
            if not isinstance(fn, (types.FunctionType, torch.jit.ScriptFunction)):
                fn_vt = VariableTracker.build(tx, fn, source=source, realize=True)
                return fn_vt.call_function(tx, [self] + list(args), kwargs)
            else:
                # Ideally we would have just used VariableTracker.build(tx, fn,
                # source=source) but that introduces guard on the
                # `forward.__code__` object. Given that we already guard on the
                # forward not present in generic dict, we don't need this guard.
                return variables.UserFunctionVariable(fn, source=source).call_function(
                    tx, [self] + list(args), kwargs
                )

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name in ["_call_impl", "_wrapped_call_impl"]:
            fn = getattr(self.value_type, name)
            if self.source:
                source = self.get_source_by_walking_mro(tx, name)
            else:
                source = None

            fn_vt = VariableTracker.build(tx, fn, source=source, realize=True)
            return fn_vt.call_function(tx, [self] + list(args), kwargs)

        if not self.has_key_in_generic_dict(tx, name):
            try:
                method = inspect.getattr_static(type(self.value), name)
            except AttributeError:
                method = None

            if isinstance(method, staticmethod):
                source = AttrSource(
                    self.get_source_by_walking_mro(tx, name), "__func__"
                )
                fn_vt = VariableTracker.build(
                    tx, method.__func__, source=source, realize=True
                )
                return fn_vt.call_function(tx, args, kwargs)

            if (
                hasattr(method, "__code__")
                and id(method.__code__) in self._nn_module_method_ids()
            ):
                unimplemented(
                    gb_type="UnspecializedNNModuleVariable missing method",
                    context=f"call_method: {self} {name} {args} {kwargs}",
                    explanation=f"Dynamo does not support tracing method {name} of nn.Module {self.value}",
                    hints=[
                        "Dynamo does not really define unspecialized nn.Module very well.",
                        *graph_break_hints.DIFFICULT,
                    ],
                )

            # "_parameters" in self.value.__dict__ checks that module is initialized
            if name == "__setattr__" and "_parameters" in self.value.__dict__:
                # Record if mutations happens on parameters/buffers/modules. The
                # mutations on these are not tracked by base class
                # UserDefinedObject vt. This will be used later to graph break
                # on seeing a parameters() and family calls.
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
                    value.is_tensor() and value.python_type() is torch.nn.Parameter
                ) or attr_name in self.value.__dict__["_parameters"]:
                    # Handle parameters
                    self.is_state_mutated = True
                elif attr_name in self.value.__dict__["_buffers"]:
                    # Handle buffers
                    self.is_state_mutated = True
                elif (
                    isinstance(value, variables.UnspecializedNNModuleVariable)
                    or attr_name in self.value.__dict__["_modules"]
                ):
                    # Handle submodules
                    self.is_state_mutated = True

            if (
                method is torch.nn.Module.__setattr__
                and isinstance(args[1], variables.DeletedVariable)
            ) or method is torch.nn.Module.__delattr__:
                # Trace through __delattr__ to track mutations on the module
                # members like `_modules``.
                fn_vt = VariableTracker.build(tx, torch.nn.Module.__delattr__)
                return fn_vt.call_function(tx, [self, args[0]], kwargs)

        return super().call_method(tx, name, list(args), kwargs)

    def getattr_helper(
        self, tx: "InstructionTranslatorBase", field: str, name_vt: VariableTracker
    ) -> VariableTracker | None:
        dict_vt = self.tp_getattro_impl(tx, field)
        if isinstance(dict_vt, variables.UserDefinedDictVariable):
            dict_vt = dict_vt._base_vt
        if isinstance(dict_vt, variables.ConstDictVariable):
            return dict_vt.maybe_getitem_const(name_vt)
        return None

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        if (
            tx.output.side_effects.is_attribute_mutation(self)
            and name
            in (
                "named_parameters",
                "parameters",
                "named_buffers",
                "buffers",
                "named_modules",
                "modules",
            )
            and self.is_state_mutated
            and tx.output.side_effects.has_pending_mutation(self)
        ):
            unimplemented(
                gb_type="getattr() on nn.Module with pending mutation",
                context=f"getattr({self}, {name})",
                explanation="Intentionally graph breaking on getattr() on a nn.Module "
                "with a pending mutation",
                hints=[],
            )

        # Allow skipping of empty hook dict guards on inbuilt nn modules
        if name in (
            "_backward_hooks",
            "_backward_pre_hooks",
            "_forward_hooks_with_kwargs",
            "_forward_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_forward_pre_hooks",
        ):
            # For empty hooks, make an EMPTY_NN_MODULE_HOOKS_DICT. This allows us to control the installation of empty
            # hooks guard via skip_nnmodule_hook_guards
            if not tx.output.side_effects.has_pending_mutation_of_attr(self, name):
                hooks_dict = getattr(self.value, name)
                if isinstance(hooks_dict, dict) and len(hooks_dict) == 0:
                    hooks_source = (
                        AttrSource(self.source, name) if self.source else None
                    )
                    if hooks_source:
                        install_guard(
                            hooks_source.make_guard(
                                GuardBuilder.EMPTY_NN_MODULE_HOOKS_DICT
                            )
                        )
                    hooks_vt_cls = (
                        variables.OrderedDictVariable
                        if isinstance(hooks_dict, collections.OrderedDict)
                        else variables.ConstDictVariable
                    )
                    return hooks_vt_cls({}, source=hooks_source)

        # For non-empty hook dicts, one way is to just fallback to VariableTracker.build() and create a ConstDictVariable.
        # However, ConstDictVariable guards on keys. This can cause recompiles when the same hook is installed for
        # different nn module instances, because the key keeps changing (look more into RemovableHandle to understand why
        # key changes - also related https://github.com/pytorch/pytorch/issues/125836). Here, we carefully craft a
        # NNModuleHooksDictVariable (a subclass of ConstDictVariable) to avoid any guard on the keys.
        if (
            self.source
            and name
            in (
                "_forward_pre_hooks_with_kwargs",
                "_forward_pre_hooks",
                "_forward_hooks_with_kwargs",
                "_forward_hooks",
            )
            and not tx.output.side_effects.has_pending_mutation_of_attr(self, name)
        ):
            hooks_dict = getattr(self.value, name)
            hooks_dict_source = AttrSource(self.source, name)
            install_guard(hooks_dict_source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            tx.output.guard_on_key_order.add(hooks_dict_source)

            def build_key_value(
                i: int, k: Any, v: Any
            ) -> tuple[VariableTracker, VariableTracker]:
                # Make key sourceless to avoid any guard on it
                key = VariableTracker.build(tx, k)

                # Instead of using dict[key] to access the value, use a dict[dict.keys()[index]] to access the
                # value. This removes the reliance on the actual key value.
                source_key = ConstDictKeySource(hooks_dict_source, i)
                source_value = DictGetItemSource(hooks_dict_source, source_key)
                value = LazyVariableTracker.create(v, source_value, tx=tx)
                return key, value

            result = dict(
                build_key_value(i, k, v)
                for i, k, v in enumerate_items_with_dict_position(hooks_dict)
            )

            return variables.NNModuleHooksDictVariable(result, source=hooks_dict_source)
        return super().tp_getattro_impl(tx, name)

    def manually_trace_nn_module_getattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        """
        Dynamo tracing of nn.Module __getattr__ can be expensive if the model
        has deep submodule hierarchy. Since the __getattr__ is stable, we can
        directly look into the underlying datastructures. This saves a lot of
        compilation time.
        """
        from .builder import SourcelessBuilder

        name_vt = SourcelessBuilder.create(tx, name)
        out = self.getattr_helper(tx, "_parameters", name_vt)
        if out is None:
            out = self.getattr_helper(tx, "_modules", name_vt)
        if out is None:
            out = self.getattr_helper(tx, "_buffers", name_vt)
        if out is None:
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(self.value).__name__}' object has no attribute '{name}'"
                ],
            )
        if out is None:
            raise AssertionError(
                f"manually_trace_nn_module_getattr failed to find attribute '{name}'"
            )
        return out


class UnspecializedBuiltinNNModuleVariable(UnspecializedNNModuleVariable):
    """
    Differentiates between builtin nn modules (e.g. torch.nn.Linear) and user defined nn modules.
    """

    def _wrap_source(self, attr_source: Source) -> Source:
        # vt is already wrapped with the UnspecializedBuiltinNNModuleSource
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

    def __init__(self, value: torch.nn.Module, **kwargs: Any) -> None:
        source = kwargs.get("source")
        if source is None:
            raise AssertionError(
                "FSDPManagedNNModule depends on having an accurate source to control guarding."
            )

        super().__init__(value=value, **kwargs)
        self.source = source

    def _wrap_source(self, attr_source: Any) -> Any:
        if not isinstance(
            attr_source, (FSDPNNModuleSource, UnspecializedNNModuleSource)
        ):
            if torch._dynamo.config.skip_fsdp_guards:
                return FSDPNNModuleSource(attr_source)
            else:
                return UnspecializedNNModuleSource(attr_source)
        return attr_source
