# mypy: ignore-errors
import dataclasses
import inspect
import sys
import warnings
from typing import Callable, Dict, List, Optional

import torch._C
from torch._guards import Guard

from .. import variables
from ..bytecode_transformation import (
    create_call_function,
    create_instruction,
    create_setup_with,
)
from ..device_interface import get_interface_for_device
from ..exc import unimplemented, Unsupported
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalStateSource
from .base import VariableTracker
from .functions import (
    NestedUserFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
    WrappedUserFunctionVariable,
    WrappedUserMethodVariable,
)
from .user_defined import UserDefinedObjectVariable


@dataclasses.dataclass
class ContextMangerState:
    """
    Mutating `self` in VariableTracker is not allowed because we copy
    them.  This is a mutable container pointed to by context managers
    that won't get copied, so it is safe to mutate.
    """

    cleanup_fn: Optional[Callable] = None
    proxy: Optional[torch.fx.Proxy] = None

    def cleanup(self):
        if self.cleanup_fn is not None:
            self.cleanup_fn()
            self.cleanup_fn = None

    def cleanup_assert(self):
        assert self.cleanup_fn, "multiple exits?"
        self.cleanup()


class ContextWrappingVariable(VariableTracker):
    _nonvar_fields = {
        "cm_obj",
        "target_values",
        "initial_values",
        "state",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, target_values, initial_values=None, *, state=None, **kwargs):
        super().__init__(**kwargs)
        self.target_values = target_values
        self.initial_values = initial_values
        self.state = ContextMangerState() if state is None else state

    def enter(self, tx):
        self._call_func(tx, self.target_values)
        self.set_cleanup_hook(tx)
        return variables.ConstantVariable.create(None)

    def set_cleanup_hook(self, tx, fn=None):
        if fn is None:

            def fn():
                self._call_func(tx, self.initial_values)

        self.state.cleanup_fn = fn
        tx.output.add_cleanup_hook(self.state.cleanup)

    def exit(self, tx, *args):
        self.state.cleanup_assert()
        return variables.ConstantVariable.create(None)

    def reconstruct_type(self, codegen):
        codegen(
            AttrSource(codegen.tx.import_source(self.module_name()), self.fn_name())
        )

    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: self.reconstruct_type(codegen))
        target_values = self.target_values
        if not target_values:
            target_values = ()
        codegen.extend_output([codegen.create_load_const(val) for val in target_values])
        codegen.extend_output(create_call_function(len(target_values), False))

    def module_name(self):
        raise NotImplementedError("module_name called on base")

    def fn_name(self):
        raise NotImplementedError("fn_name called on base")

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        assert len(args) == 1
        if isinstance(args[0], NestedUserFunctionVariable):
            args[0] = UserFunctionVariable(args[0].get_function())
        assert isinstance(args[0], (UserMethodVariable, UserFunctionVariable))

        if isinstance(args[0], UserMethodVariable):
            return WrappedUserMethodVariable(args[0], self)

        if isinstance(args[0], UserFunctionVariable):
            return WrappedUserFunctionVariable(args[0], self)


class GenericContextWrappingVariable(
    ContextWrappingVariable, UserDefinedObjectVariable
):
    # Some methods in ContextWrappingVariable assumes the arguments are
    # python contants. Which might not always be the case here.
    def __init__(self, cm_obj, **kwargs):
        assert cm_obj is not None
        super().__init__(
            target_values=None,
            initial_values=None,
            value=cm_obj,
            value_type=cm_obj.__class__,
            **kwargs,
        )
        self.cm_obj = cm_obj

    def module_name(self):
        return self.cm_obj.__module__

    def fn_name(self):
        return type(self.cm_obj).__name__

    def reconstruct(self, codegen):
        # throw NotImplementedError
        super(UserDefinedObjectVariable, self).reconstruct(codegen)

    def enter(self, tx):
        source = None if self.source is None else AttrSource(self.source, "__enter__")
        try:
            return variables.UserMethodVariable(
                self.cm_obj.__enter__.__func__,
                self,
                source=source,
            ).call_function(tx, [], {})
        except Unsupported as e:
            unimplemented(
                f"Unsupported context manager {self.cm_obj}'s __enter__ function",
                from_exc=e,
            )

    def exit(self, tx, *args):
        source = None if self.source is None else AttrSource(self.source, "__exit__")
        try:
            x = variables.UserMethodVariable(
                self.cm_obj.__exit__.__func__,
                self,
                source=source,
            ).call_function(
                tx,
                [
                    variables.ConstantVariable.create(None),
                    variables.ConstantVariable.create(None),
                    variables.ConstantVariable.create(None),
                ],
                {},
            )
        except Unsupported as e:
            unimplemented(
                f"Unsupported context manager {self.cm_obj}'s __exit__ function",
                from_exc=e,
            )

        tx.generic_context_manager_depth -= 1
        return x


class GradInplaceRequiresGradCtxManagerVariable(ContextWrappingVariable):
    """represents torch grad requries grad"""

    @staticmethod
    def create(tx, target_values, **kwargs):
        return GradInplaceRequiresGradCtxManagerVariable(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx):
        [enabled] = self.target_values
        self.prev_state = torch._C._functorch.get_inplace_requires_grad_allowed()
        torch._C._functorch.set_inplace_requires_grad_allowed(enabled)
        self.set_cleanup_hook(
            tx,
            lambda: torch._C._functorch.set_inplace_requires_grad_allowed(
                self.prev_state
            ),
        )
        self.state.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch.set_inplace_requires_grad_allowed,
            (enabled,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(self, tx, *args):
        self.state.cleanup()
        tx.output.create_node(
            "call_function",
            torch._C._functorch.set_inplace_requires_grad_allowed,
            (self.prev_state,),
            {},
        )
        return variables.ConstantVariable.create(None)


class JvpIncrementNestingCtxManagerVariable(ContextWrappingVariable):
    """represents torch.func.jvp increment/decrement nesting"""

    # A guard is needed as the grad level is baked into the torch FX graph
    # This is fine if jvp is only called from within the function
    # being compiled. But the FX graph may be invalid in the case of a jvp
    # call from eager that calls the compiled function, as the jvp levels
    # may be different.
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FUNCTORCH_STACK_MATCH)

    @staticmethod
    def create(tx, **kwargs):
        var = JvpIncrementNestingCtxManagerVariable(
            target_values=None,
            initial_values=None,
            **kwargs,
        )
        return var

    def enter(self, tx):
        install_guard(self._guards_singleton)
        jvp_level = torch._functorch.eager_transforms.enter_jvp_nesting()
        self.set_cleanup_hook(
            tx, lambda: torch._functorch.eager_transforms.exit_jvp_nesting()
        )
        self.state.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch._jvp_increment_nesting,
            (),
            {},
        )
        return variables.ConstantVariable.create(jvp_level)

    def exit(self, tx, *args):
        self.state.cleanup()
        tx.output.create_node(
            "call_function", torch._C._functorch._jvp_decrement_nesting, (), {}
        )
        return variables.ConstantVariable.create(None)


class SetFwdGradEnabledContextManager(ContextWrappingVariable):
    """represents torch.autograd.forward_ad._set_fwd_grad_enabled() to enable/disable fwd grad"""

    @staticmethod
    def create(tx, target_values, **kwargs):
        return SetFwdGradEnabledContextManager(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx):
        [mode] = self.target_values
        self.prev_state = torch._C._is_fwd_grad_enabled()
        torch._C._set_fwd_grad_enabled(mode)
        self.set_cleanup_hook(
            tx,
            lambda: torch._C._set_fwd_grad_enabled(self.prev_state),
        )
        self.state.proxy = tx.output.create_node(
            "call_function",
            torch._C._set_fwd_grad_enabled,
            (mode,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(self, tx, *args):
        self.state.cleanup()
        tx.output.create_node(
            "call_function",
            torch._C._set_fwd_grad_enabled,
            (self.prev_state,),
            {},
        )
        return variables.ConstantVariable.create(None)


class DualLevelContextManager(ContextWrappingVariable):
    """Represents torch.autograd.forward_ad.dual_level ctx manager"""

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.DUAL_LEVEL)

    @staticmethod
    def create(tx, **kwargs):
        return DualLevelContextManager(
            target_values=None,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx):
        install_guard(self._guards_singleton)
        self.new_level = torch.autograd.forward_ad.enter_dual_level()
        self.set_cleanup_hook(
            tx, lambda: torch.autograd.forward_ad.exit_dual_level(level=self.new_level)
        )
        self.state.proxy = tx.output.create_node(
            "call_function",
            torch._C._enter_dual_level,
            (),
            {},
        )
        return variables.ConstantVariable.create(self.new_level)

    def exit(self, tx, *args):
        self.state.cleanup()
        tx.output.create_node(
            "call_function",
            torch._C._exit_dual_level,
            (self.new_level,),
            {},
        )
        return variables.ConstantVariable.create(None)


class GradIncrementNestingCtxManagerVariable(ContextWrappingVariable):
    """represents torch.func.grad increment/decrement nesting"""

    # A guard is needed as the grad level is baked into the torch FX graph
    # This is fine if grad is only called from within the function
    # being compiled. But the FX graph may be invalid in the case of a grad
    # call from eager that calls the compiled function, as the grad levels
    # may be different.
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FUNCTORCH_STACK_MATCH)

    @staticmethod
    def create(tx, **kwargs):
        var = GradIncrementNestingCtxManagerVariable(
            target_values=None,
            initial_values=None,
            **kwargs,
        )
        return var

    def enter(self, tx):
        install_guard(self._guards_singleton)
        grad_level = torch._C._functorch._grad_increment_nesting()
        self.set_cleanup_hook(tx, lambda: torch._C._functorch._grad_decrement_nesting())
        self.state.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch._grad_increment_nesting,
            (),
            {},
        )
        return variables.ConstantVariable.create(grad_level)

    def exit(self, tx, *args):
        self.state.cleanup()
        tx.output.create_node(
            "call_function", torch._C._functorch._grad_decrement_nesting, (), {}
        )
        return variables.ConstantVariable.create(None)


class CatchWarningsCtxManagerVariable(ContextWrappingVariable):
    """Delay a call to warnings.catch_warnings"""

    @staticmethod
    def create(tx, catch_warnings_args):
        return CatchWarningsCtxManagerVariable(
            catch_warnings_args=catch_warnings_args,
            target_values=None,
            initial_values=None,
        )

    def __init__(self, catch_warnings_args, **kwargs):
        assert isinstance(catch_warnings_args, dict), catch_warnings_args
        super().__init__(**kwargs)
        self.catch_warnings_args = catch_warnings_args

    def enter(self, tx):
        kwargs = {
            k: v.as_python_constant() for k, v in self.catch_warnings_args.items()
        }
        ctx_val = warnings.catch_warnings(**kwargs)
        self.set_cleanup_hook(tx, lambda: ctx_val.__exit__(None, None, None))
        return variables.ConstantVariable.create(ctx_val.__enter__())

    def reconstruct(self, cg):
        cg.add_push_null(lambda: cg.load_import_from("warnings", "catch_warnings"))
        cg.foreach(self.catch_warnings_args.values())
        keys = tuple(self.catch_warnings_args.keys())
        cg.extend_output(cg.create_call_function_kw(len(keys), keys, False))


class VmapIncrementNestingCtxManagerVariable(ContextWrappingVariable):
    """represents torch VMap increment/decrement nesting"""

    # A guard is needed as the vmap level is baked into the torch FX graph
    # generated. This is fine if vmap is only called from within the function
    # being compiled. But the FX graph may be invalid in the case of a vmap
    # call from eager that calls the compiled function, as the vmap levels
    # may be different.
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FUNCTORCH_STACK_MATCH)

    @staticmethod
    def create(tx, target_values, **kwargs):
        var = VmapIncrementNestingCtxManagerVariable(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )
        return var

    def enter(self, tx):
        install_guard(self._guards_singleton)
        batch_size, randomness = self.target_values
        vmap_level = torch._C._functorch._vmap_increment_nesting(batch_size, randomness)
        self.set_cleanup_hook(tx, lambda: torch._C._functorch._vmap_decrement_nesting())
        self.state.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch._vmap_increment_nesting,
            (batch_size, randomness),
            {},
        )
        return variables.ConstantVariable.create(vmap_level)

    def exit(self, tx, *args):
        self.state.cleanup()
        tx.output.create_node(
            "call_function", torch._C._functorch._vmap_decrement_nesting, (), {}
        )
        return variables.ConstantVariable.create(None)


class GradModeVariable(ContextWrappingVariable):
    """represents torch.{no_grad,enable_grad,set_grad_mode}()"""

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.GRAD_MODE)

    @staticmethod
    def create(tx, target_value, initialized=False, **kwargs):
        var = GradModeVariable(
            target_values=[target_value],
            initial_values=[torch.is_grad_enabled()],
            **kwargs,
        )
        if initialized:
            var._call_func(tx, var.target_values)
        return var

    def __init__(self, target_values, initial_values=None, initialized=True, **kwargs):
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        install_guard(self._guards_singleton)

    def enter(self, tx):
        self._call_func(tx, self.target_values)
        return variables.ConstantVariable.create(None)

    def exit(self, tx, *args):
        self._call_func(tx, self.initial_values)
        return variables.ConstantVariable.create(None)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ):
        self._call_func(tx, self.initial_values)  # undo eager initialization
        return super().call_function(tx, args, kwargs)

    def _call_func(self, tx, values):
        assert len(values) == 1
        value = values[0]
        # Coalesce grad mode mutations
        if torch.is_grad_enabled() != value:
            tx.output.create_node(
                "call_function", torch._C._set_grad_enabled, (value,), {}
            )
            torch._C._set_grad_enabled(value)

    def module_name(self):
        return "torch"

    def fn_name(self):
        return "set_grad_enabled"


class InferenceModeVariable(ContextWrappingVariable):
    @staticmethod
    def create(tx, target_value, **kwargs):
        var = InferenceModeVariable(
            [target_value], initial_values=torch.is_inference_mode_enabled(), **kwargs
        )
        return var

    def __init__(
        self,
        target_values,
        initial_values=None,
        **kwargs,
    ):
        if initial_values is None:
            # This must be called here since function defaults are evaluated at import time
            initial_values = torch.is_inference_mode_enabled()
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.target_values = target_values

    def exit(self, tx, *args):
        self.state.cleanup_assert()
        tx.output.create_node(
            "call_function",
            torch.autograd.grad_mode._exit_inference_mode,
            (self.state.proxy,),
            {},
        )

    def enter(self, tx):
        ctx = torch.autograd.grad_mode._enter_inference_mode(*self.target_values)
        self.set_cleanup_hook(
            tx, lambda: torch.autograd.grad_mode._exit_inference_mode(ctx)
        )
        self.state.proxy = tx.output.create_node(
            "call_function",
            torch.autograd.grad_mode._enter_inference_mode,
            (*self.target_values,),
            {},
        )

    def module_name(self):
        return "torch"

    def fn_name(self):
        return "inference_mode"


class TorchFunctionDisableVariable(ContextWrappingVariable):
    """represents whether torch function overrides are enabled or not"""

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.TORCH_FUNCTION_STATE)

    @staticmethod
    def create(tx, **kwargs):
        var = TorchFunctionDisableVariable(
            target_values=[False],
            initial_values=[tx.output.torch_function_enabled],
            **kwargs,
        )
        # mlazos: I think this is here to make sure we don't reinvoke on clone()
        var._call_func(tx, [False])
        var.set_cleanup_hook(tx)
        return var

    def __init__(self, target_values, initial_values=None, **kwargs):
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        install_guard(self._guards_singleton)

    def enter(self, tx):
        return variables.ConstantVariable.create(None)

    def _call_func(self, tx, values):
        assert len(values) == 1
        tx.output.set_torch_function_state(values[0])


class DeterministicAlgorithmsVariable(ContextWrappingVariable):
    """represents torch.{are_deterministic_algorithms_enabled,use_deterministic_algorithms}()"""

    _guards_singleton = Guard(
        GlobalStateSource(), GuardBuilder.DETERMINISTIC_ALGORITHMS
    )

    @staticmethod
    def create(tx, target_value, **kwargs):
        var = DeterministicAlgorithmsVariable(
            target_values=[target_value],
            initial_values=[torch.are_deterministic_algorithms_enabled()],
            **kwargs,
        )
        var._call_func(tx, [target_value])
        var.set_cleanup_hook(tx)
        return var

    def __init__(self, target_values, initial_values=None, **kwargs):
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        install_guard(self._guards_singleton)

    def enter(self, tx):
        return variables.ConstantVariable.create(None)

    def _call_func(self, tx, values):
        assert len(values) == 1
        value = values[0]
        tx.output.create_node(
            "call_function", torch._C._set_deterministic_algorithms, (value,), {}
        ),
        torch._C._set_deterministic_algorithms(value)

    def module_name(self):
        return "torch"

    def fn_name(self):
        return "use_deterministic_algorithms"


class DisabledSavedTensorsHooksVariable(ContextWrappingVariable):
    """represents torch.autograd.graph.disable_saved_tensors_hook."""

    @staticmethod
    def create(tx, target_value, **kwargs):
        var = DisabledSavedTensorsHooksVariable(
            target_values=[target_value],
            initial_values=[
                torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()
            ],
            **kwargs,
        )
        var._call_func(tx, [target_value])
        var.set_cleanup_hook(tx)
        return var

    def __init__(self, target_values, initial_values=None, **kwargs):
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def enter(self, tx):
        return variables.ConstantVariable.create(None)

    def _call_func(self, tx, values):
        assert len(values) == 1
        value = values[0]
        if value is not None:
            # Disable `saved_tensors_hooks` with message (`value`)
            # OR
            # we are exiting this context and restoring the previous message.
            tx.output.create_node(
                "call_function",
                torch._C._autograd._saved_tensors_hooks_disable,
                (value,),
                {},
            )
            torch._C._autograd._saved_tensors_hooks_disable(value)
        else:
            # We are exiting this context and if prev_message was None, we re-enable `saved_tensors_hooks`.
            tx.output.create_node(
                "call_function", torch._C._autograd._saved_tensors_hooks_enable, (), {}
            )
            torch._C._autograd._saved_tensors_hooks_enable()

    def module_name(self):
        return "torch.autograd.graph"

    def fn_name(self):
        return "disable_saved_tensors_hooks"


class AutocastModeVariable(ContextWrappingVariable):
    @staticmethod
    def create(func, args, kwargs):
        assert func in [
            torch.amp.autocast_mode.autocast,
            torch.cuda.amp.autocast,
            torch.cpu.amp.autocast,
        ]
        # device_type : str,
        # dtype : Optional[_dtype] = None,
        # enabled : bool = True,
        # cache_enabled : Optional[bool] = None):cache_enabled
        bound_args = inspect.signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()
        target_values = []
        kwargs.clear()

        for key in ["device_type", "dtype", "enabled", "cache_enabled"]:
            if key == "device_type" and func in [
                torch.cuda.amp.autocast,
                torch.cpu.amp.autocast,
            ]:
                arg = "cuda" if func is torch.cuda.amp.autocast else "cpu"
            else:
                arg = bound_args.arguments[key]
            if isinstance(arg, VariableTracker):
                target_values.append(arg.as_python_constant())
            else:
                target_values.append(arg)

        var = AutocastModeVariable(target_values, initial_values=None, **kwargs)
        return var

    def __init__(self, target_values, initial_values=None, **kwargs):
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.target_values = target_values

    def exit(self, tx, *args):
        self.state.cleanup_assert()
        tx.output.create_node(
            "call_function", torch.amp._exit_autocast, (self.state.proxy,), {}
        )

    def enter(self, tx):
        ctx = torch.amp._enter_autocast(*self.target_values)
        self.set_cleanup_hook(tx, lambda: torch.amp._exit_autocast(ctx))
        self.state.proxy = tx.output.create_node(
            "call_function", torch.amp._enter_autocast, (*self.target_values,), {}
        )

    def module_name(self):
        return "torch.amp.autocast_mode"

    def fn_name(self):
        return "autocast"


class NullContextVariable(ContextWrappingVariable):
    """
    This class represents Python contextlib.nullcontext.
    It's used as a placeholder for other context managers that Dynamo doesn't
    support yet, e.g, torch.autograd.profiler.record_function.
    """

    def __init__(self, target_values=None, **kwargs):
        super().__init__(target_values=target_values, **kwargs)

    def enter(self, tx):
        return variables.ConstantVariable.create(None)

    def exit(self, tx, *args):
        return variables.ConstantVariable.create(None)

    def module_name(self):
        return "contextlib"

    def fn_name(self):
        return "nullcontext"


class StreamContextVariable(ContextWrappingVariable):
    @staticmethod
    def create(tx, target_value, **kwargs):
        from .builder import wrap_fx_proxy_cls

        current_stream_method = get_interface_for_device(
            target_value.device
        ).current_stream
        current_stream = wrap_fx_proxy_cls(
            StreamVariable,
            tx,
            tx.output.create_proxy(
                "call_function",
                current_stream_method,
                (None,),
                {},
            ),
        )
        return StreamContextVariable(
            target_values=[target_value],
            initial_values=[current_stream],
            device=target_value.device,
            **kwargs,
        )

    def __init__(self, target_values, device, initial_values=None, **kwargs):
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.device = device
        self.set_stream = get_interface_for_device(self.device).set_stream
        self.set_stream_id = get_interface_for_device(self.device)._set_stream_by_id

    def enter(self, tx):
        # stream generated inside the traced function
        if self.target_values[0].as_proxy() is not None:
            tx.output.create_proxy(
                "call_function",
                self.set_stream,
                (self.target_values[0].as_proxy(),),
                {},
            )
        # stream passed from outside the traced function
        else:
            stream = self.target_values[0].value
            tx.output.create_proxy(
                "call_function",
                self.set_stream_id,
                (stream.stream_id, stream.device_index, stream.device_type),
                {},
            )
        self.set_stream(self.target_values[0].value)
        self.set_cleanup_hook(tx, lambda: self.set_stream(self.initial_values[0].value))

    def exit(self, tx, *args):
        tx.output.create_proxy(
            "call_function",
            self.set_stream,
            (self.initial_values[0].as_proxy(),),
            {},
        )
        self.state.cleanup_assert()


class PreserveVersionContextVariable(ContextWrappingVariable):
    """
    Wraps torch.autograd._unsafe_preserve_version_counter
    """

    @staticmethod
    def constructor(tx):
        return variables.LambdaVariable(
            lambda tensor: PreserveVersionContextVariable(
                tensor,
                tensor.var_getattr(tx, "_version"),
            )
        )

    def __init__(self, tensor, prev_version, **kwargs):
        kwargs.setdefault("target_values", None)
        super().__init__(**kwargs)
        self.tensor = tensor
        self.prev_version = prev_version

    def enter(self, tx):
        pass

    def exit(self, tx, *args):
        from ..tensor_version_op import _unsafe_set_version_counter

        return variables.TorchInGraphFunctionVariable(
            _unsafe_set_version_counter
        ).call_function(tx, [self.tensor, self.prev_version], {})

    def reconstruct(self, codegen):
        unimplemented(
            "torch.autograd._unsafe_preserve_version_counter with graph break"
        )


class FSDPParamGroupUseTrainingStateVariable(ContextWrappingVariable):
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FSDP_TRAINING_STATE)

    @staticmethod
    def create(tx, param_group_var, target_value, **kwargs):
        var = FSDPParamGroupUseTrainingStateVariable(
            param_group_var=param_group_var,
            target_values=[target_value],
            initial_values=[param_group_var.value._training_state],
            **kwargs,
        )
        return var

    def __init__(self, param_group_var, target_values, initial_values=None, **kwargs):
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.param_group_var = param_group_var
        install_guard(self._guards_singleton)

    def enter(self, tx):
        self._call_func(tx, self.target_values)
        return variables.ConstantVariable.create(None)

    def exit(self, tx, *args):
        self._call_func(tx, self.initial_values)
        return variables.ConstantVariable.create(None)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ):
        self._call_func(tx, self.initial_values)  # undo eager initialization
        return super().call_function(tx, args, kwargs)

    def _call_func(self, tx, values):
        assert len(values) == 1
        value = values[0]
        if self.param_group_var.value._training_state != value:
            self.param_group_var.call_method(
                tx,
                "__setattr__",
                (
                    variables.ConstantVariable.create("_training_state"),
                    variables.EnumVariable(value),
                ),
                {},
            )
            self.param_group_var.value._training_state = value

    def module_name(self):
        return "torch.distributed._composable.fsdp._fsdp_param_group.FSDPParamGroup"

    def fn_name(self):
        return "use_training_state"


class StreamVariable(VariableTracker):
    def __init__(self, proxy, value, device, **kwargs):
        if proxy is not None and "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == value
        assert (
            value.device.type == device.type
        ), "stream value is not equal to the passed device"
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.device = device

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        assert hasattr(self.value, name), f"no stream method found named {name}"
        assert name in [
            "wait_stream",
            "synchronize",
            "query",
            "record_event",
            "wait_event",
        ], f" unsupported stream method {name}"

        from ..utils import proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls

        if name in ("wait_stream", "synchronize", "wait_event"):
            tx.output.create_proxy(
                "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
            )
            return variables.ConstantVariable(None)
        elif name == "query":
            return wrap_fx_proxy_cls(
                target_cls=variables.ConstantVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
                ),
            )
        elif name == "record_event":
            return wrap_fx_proxy_cls(
                target_cls=EventVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
                ),
            )
        else:
            unimplemented(self.device + " stream method " + name + " unsupported")

    def as_proxy(self):
        return self.proxy

    def reconstruct(self, codegen):
        # If we got here, this stream is fully subsumed by the graph - this means it is
        # not an input or global
        assert not self.source
        # Since we just proved that - for other such structures, like lists and dicts, reconstruction
        # is fine and sound according to dynamo principles of treating collectives. However,
        # streams are special in that we want to preserve the identity of the stream as the same as in the graph
        # Normally, we would do this via codegen for the proxy mapping to an output - we cannot do this yet, as we do not
        # yet have a plan for how we want to handle the case where the stream is used as an input or an output. Pending
        # design, to unblock current work, we lift the stream into a global and then codegen bytecode to load it from there.
        prefix = f"_stream_{self.device}"
        name = codegen.tx.output.install_global_by_id(prefix, self.value)
        codegen.append_output(codegen.create_load_global(name, add=True))


class EventVariable(VariableTracker):
    def __init__(self, proxy, value, **kwargs):
        if proxy is not None and "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == value
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from ..utils import proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls

        if name in ("wait", "record", "synchronize"):
            tx.output.create_proxy(
                "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
            )
            return variables.ConstantVariable(None)
        elif name == "query":
            return wrap_fx_proxy_cls(
                target_cls=variables.ConstantVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
                ),
            )
        else:
            unimplemented(f"event method {name} unsupported")

    def as_proxy(self):
        return self.proxy


class WithExitFunctionVariable(VariableTracker):
    _nonvar_fields = {
        "target",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, ctx: ContextWrappingVariable, target, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(ctx, ContextWrappingVariable)
        self.ctx = ctx
        self.target = target

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        assert not kwargs
        return self.ctx.exit(tx, *args)

    def reconstruct(self, codegen):
        # Note here we reconstruct the context manager rather than the
        # exit function.  The handler generated by BlockStackEntry
        # will re-enter the context in the resume function.
        self.ctx.reconstruct_type(codegen)
        if codegen.tx.output.partial_convert:
            if sys.version_info >= (3, 11):
                codegen.append_output(create_instruction("PUSH_NULL"))
                if sys.version_info < (3, 13):
                    codegen.append_output(create_instruction("SWAP", arg=2))
            codegen.extend_output(
                [codegen.create_load_const(val) for val in self.ctx.target_values]
            )
            codegen.extend_output(
                create_call_function(len(self.ctx.target_values), False)
            )
            codegen.append_output(create_setup_with(self.target))
            codegen.append_output(create_instruction("POP_TOP"))
