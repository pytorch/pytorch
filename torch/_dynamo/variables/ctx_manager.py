"""
This file contains a collection of context manager classes used by Dynamo for tracking
and managing various PyTorch runtime states during graph compilation. These context
managers handle different aspects of PyTorch's execution environment, including:

- Autograd states (grad mode, inference mode)
- CUDA streams and events
- Profiling contexts
- Deterministic algorithms
- Forward/backward AD modes
- SDPA (Scaled Dot Product Attention) kernels
- FSDP (Fully Sharded Data Parallel) states
- AMP (Automatic Mixed Precision) autocast states

The context managers ensure proper state transitions during graph compilation by
tracking enter/exit points and managing cleanup operations. They help maintain
consistency between eager execution and compiled graph behavior by capturing and
restoring state changes.
"""

import inspect
import sys
import warnings
from collections.abc import Callable, Sequence, Sized
from contextlib import AbstractContextManager, ExitStack
from typing import Any, Optional, TYPE_CHECKING, Union

import torch._C
from torch._guards import Guard

from .. import graph_break_hints, variables
from ..bytecode_transformation import (
    create_call_function,
    create_instruction,
    create_setup_with,
)
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalStateSource
from ..utils import _get_error_on_graph_break, _set_error_on_graph_break
from .base import VariableTracker
from .functions import (
    NestedUserFunctionVariable,
    SkipFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
    WrappedNestedUserFunctionVariable,
    WrappedSkipFunctionVariable,
    WrappedUserFunctionVariable,
    WrappedUserMethodVariable,
)
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator


class ContextWrappingVariable(VariableTracker):
    _nonvar_fields = {
        "cm_obj",
        "target_values",
        "initial_values",
        "state",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self, target_values: Any, initial_values: Optional[Any] = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.target_values = target_values
        self.initial_values = initial_values

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        if hasattr(self, "_call_func"):
            self._call_func(tx, self.target_values)
        self.set_cleanup_hook(tx)
        return variables.ConstantVariable.create(None)

    def set_cleanup_hook(
        self, tx: "InstructionTranslator", fn: Optional[Callable[..., Any]] = None
    ) -> None:
        if fn is None:

            def fn() -> None:
                if hasattr(self, "_call_func"):
                    self._call_func(tx, self.initial_values)

        self.cleanup_fn: Optional[Callable[..., Any]] = fn
        tx.output.add_cleanup_hook(self.cleanup)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup_assert()
        return variables.ConstantVariable.create(None)

    def reconstruct_type(self, codegen: "PyCodegen") -> None:
        codegen(
            AttrSource(codegen.tx.import_source(self.module_name()), self.fn_name())
        )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: self.reconstruct_type(codegen))
        target_values = self.target_values
        if not target_values:
            target_values = ()
        codegen.extend_output([codegen.create_load_const(val) for val in target_values])
        codegen.extend_output(create_call_function(len(target_values), False))

    def module_name(self) -> str:
        raise NotImplementedError("module_name called on base")

    def fn_name(self) -> str:
        raise NotImplementedError("fn_name called on base")

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        assert len(args) == 1
        assert isinstance(
            args[0],
            (
                NestedUserFunctionVariable,
                SkipFunctionVariable,
                UserMethodVariable,
                UserFunctionVariable,
            ),
        )

        if isinstance(args[0], NestedUserFunctionVariable):
            return WrappedNestedUserFunctionVariable(args[0], self)
        elif isinstance(args[0], SkipFunctionVariable):
            return WrappedSkipFunctionVariable(args[0], self)
        elif isinstance(args[0], UserMethodVariable):
            return WrappedUserMethodVariable(args[0], self)
        elif isinstance(args[0], UserFunctionVariable):
            return WrappedUserFunctionVariable(args[0], self)
        else:
            raise AssertionError("Unexpected arg type")

    def supports_graph_breaks(self) -> bool:
        return True

    def exit_on_graph_break(self) -> bool:
        return True

    def cleanup(self) -> None:
        if self.cleanup_fn is not None:
            self.cleanup_fn()
            self.cleanup_fn = None

    def cleanup_assert(self) -> None:
        assert self.cleanup_fn, "multiple exits?"
        self.cleanup()


class GenericContextWrappingVariable(UserDefinedObjectVariable):
    # Some methods in ContextWrappingVariable assumes the arguments are
    # python constants. Which might not always be the case here.
    def __init__(self, cm_obj: AbstractContextManager[Any], **kwargs: Any) -> None:
        assert cm_obj is not None
        super().__init__(
            value=cm_obj,
            value_type=cm_obj.__class__,
            **kwargs,
        )
        self.cm_obj = cm_obj

    def module_name(self) -> str:
        return self.cm_obj.__module__

    def fn_name(self) -> str:
        return type(self.cm_obj).__name__

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        source = None if self.source is None else AttrSource(self.source, "__enter__")
        return variables.UserMethodVariable(
            self.cm_obj.__enter__.__func__,  # type: ignore[attr-defined]
            self,
            source=source,
        ).call_function(tx, [], {})

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        source = None if self.source is None else AttrSource(self.source, "__exit__")
        x = variables.UserMethodVariable(
            self.cm_obj.__exit__.__func__,  # type: ignore[attr-defined]
            self,
            source=source,
        ).call_function(tx, list(args), {})
        tx.active_generic_context_managers.pop()
        return x

    def supports_graph_breaks(self) -> bool:
        return False

    def exit_on_graph_break(self) -> bool:
        return True


class RepararametrizeModuleContextVariable(GenericContextWrappingVariable):
    def __init__(self, ctx_manager_vt: ContextWrappingVariable, mod: Any) -> None:
        self.cm_vt = ctx_manager_vt
        self.mod = mod
        # We don't call super().__init__() because we're delegating most methods to cm_vt

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        # Custom enter implementation with side effects

        self.old_parameters_var = self.mod.var_getattr(tx, "_parameters").realize()
        self.old_buffer_var = self.mod.var_getattr(tx, "_buffers").realize()
        tx.output.side_effects.ignore_mutations_on(self.old_parameters_var)
        tx.output.side_effects.ignore_mutations_on(self.old_buffer_var)
        return self.cm_vt.enter(tx)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        # Custom exit implementation with side effects
        x = self.cm_vt.exit(tx, *args)
        tx.output.side_effects.stop_ignoring_mutations_on(self.old_buffer_var)
        tx.output.side_effects.stop_ignoring_mutations_on(self.old_parameters_var)
        return x

    # Forward all other method calls to self.cm_vt
    def __getattr__(self, name: str) -> Any:
        # This will be called for any attribute not explicitly defined in this class
        return getattr(self.cm_vt, name)


class GradInplaceRequiresGradCtxManagerVariable(ContextWrappingVariable):
    """represents torch grad requires grad"""

    @staticmethod
    def create(
        tx: "InstructionTranslator", target_values: Any, **kwargs: Any
    ) -> "GradInplaceRequiresGradCtxManagerVariable":
        return GradInplaceRequiresGradCtxManagerVariable(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        [enabled] = self.target_values
        self.prev_state = torch._C._functorch.get_inplace_requires_grad_allowed()
        torch._C._functorch.set_inplace_requires_grad_allowed(enabled)
        self.set_cleanup_hook(
            tx,
            lambda: torch._C._functorch.set_inplace_requires_grad_allowed(
                self.prev_state
            ),
        )
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch.set_inplace_requires_grad_allowed,
            (enabled,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function",
            torch._C._functorch.set_inplace_requires_grad_allowed,
            (self.prev_state,),
            {},
        )
        return variables.ConstantVariable.create(None)


class TemporarilyPopInterpreterStackCtxManagerVariable(ContextWrappingVariable):
    """represents torch._functorch.pyfunction.temporarily_pop_interpreter_stack()"""

    @staticmethod
    def create(
        tx: "InstructionTranslator", target_values: Any, **kwargs: Any
    ) -> "TemporarilyPopInterpreterStackCtxManagerVariable":
        return TemporarilyPopInterpreterStackCtxManagerVariable(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        self.saved = torch._C._functorch.pop_dynamic_layer_stack()
        self.set_cleanup_hook(
            tx,
            lambda: torch._C._functorch.push_dynamic_layer_stack(self.saved),
        )
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch.pop_dynamic_layer_stack,
            (),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function",
            torch._C._functorch.push_dynamic_layer_stack,
            (self.proxy,),
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
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FUNCTORCH_STACK_MATCH)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslator", **kwargs: Any
    ) -> "JvpIncrementNestingCtxManagerVariable":
        var = JvpIncrementNestingCtxManagerVariable(
            target_values=None,
            initial_values=None,
            **kwargs,
        )
        return var

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        install_guard(self._guards_singleton)
        jvp_level = torch._functorch.eager_transforms.enter_jvp_nesting()
        self.set_cleanup_hook(
            tx, lambda: torch._functorch.eager_transforms.exit_jvp_nesting()
        )
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch._jvp_increment_nesting,
            (),
            {},
        )
        return variables.ConstantVariable.create(jvp_level)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function", torch._C._functorch._jvp_decrement_nesting, (), {}
        )
        return variables.ConstantVariable.create(None)


class SetFwdGradEnabledContextManager(ContextWrappingVariable):
    """represents torch.autograd.forward_ad._set_fwd_grad_enabled() to enable/disable fwd grad"""

    @staticmethod
    def create(
        tx: "InstructionTranslator", target_values: Any, **kwargs: Any
    ) -> "SetFwdGradEnabledContextManager":
        return SetFwdGradEnabledContextManager(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        [mode] = self.target_values
        self.prev_state = torch._C._is_fwd_grad_enabled()
        torch._C._set_fwd_grad_enabled(mode)
        self.set_cleanup_hook(
            tx,
            lambda: torch._C._set_fwd_grad_enabled(self.prev_state),
        )
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._set_fwd_grad_enabled,
            (mode,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function",
            torch._C._set_fwd_grad_enabled,
            (self.prev_state,),
            {},
        )
        return variables.ConstantVariable.create(None)


class DualLevelContextManager(ContextWrappingVariable):
    """Represents torch.autograd.forward_ad.dual_level ctx manager"""

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.DUAL_LEVEL)  # type: ignore[arg-type]

    @staticmethod
    def create(tx: "InstructionTranslator", **kwargs: Any) -> "DualLevelContextManager":
        return DualLevelContextManager(
            target_values=None,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        install_guard(self._guards_singleton)
        self.new_level = torch.autograd.forward_ad.enter_dual_level()
        self.set_cleanup_hook(
            tx, lambda: torch.autograd.forward_ad.exit_dual_level(level=self.new_level)
        )
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._enter_dual_level,
            (),
            {},
        )
        return variables.ConstantVariable.create(self.new_level)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
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
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FUNCTORCH_STACK_MATCH)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslator", **kwargs: Any
    ) -> "GradIncrementNestingCtxManagerVariable":
        var = GradIncrementNestingCtxManagerVariable(
            target_values=None,
            initial_values=None,
            **kwargs,
        )
        return var

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        install_guard(self._guards_singleton)
        grad_level = torch._C._functorch._grad_increment_nesting()
        self.set_cleanup_hook(tx, lambda: torch._C._functorch._grad_decrement_nesting())
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch._grad_increment_nesting,
            (),
            {},
        )
        return variables.ConstantVariable.create(grad_level)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function", torch._C._functorch._grad_decrement_nesting, (), {}
        )
        return variables.ConstantVariable.create(None)


class CatchWarningsCtxManagerVariable(ContextWrappingVariable):
    """Delay a call to warnings.catch_warnings"""

    @staticmethod
    def create(
        tx: "InstructionTranslator", catch_warnings_args: dict[str, VariableTracker]
    ) -> "CatchWarningsCtxManagerVariable":
        return CatchWarningsCtxManagerVariable(
            catch_warnings_args=catch_warnings_args,
            target_values=None,
            initial_values=None,
        )

    def __init__(
        self,
        catch_warnings_args: dict[str, VariableTracker],
        target_values: Optional[Any] = None,
        initial_values: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        assert isinstance(catch_warnings_args, dict), catch_warnings_args
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.catch_warnings_args = catch_warnings_args

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        kwargs = {
            k: v.as_python_constant() for k, v in self.catch_warnings_args.items()
        }
        ctx_val = warnings.catch_warnings(**kwargs)
        self.set_cleanup_hook(tx, lambda: ctx_val.__exit__(None, None, None))
        return variables.ConstantVariable.create(ctx_val.__enter__())

    def reconstruct(self, cg: "PyCodegen") -> None:
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
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FUNCTORCH_STACK_MATCH)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslator",
        target_values: Sequence[VariableTracker],
        **kwargs: Any,
    ) -> "VmapIncrementNestingCtxManagerVariable":
        var = VmapIncrementNestingCtxManagerVariable(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )
        return var

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        install_guard(self._guards_singleton)
        batch_size, randomness = self.target_values
        if isinstance(batch_size, variables.SymNodeVariable):
            batch_size_value = batch_size.sym_num
        else:
            batch_size_value = batch_size.as_python_constant()
        randomness = randomness.as_python_constant()
        vmap_level = torch._C._functorch._vmap_increment_nesting(
            batch_size_value, randomness
        )
        self.set_cleanup_hook(tx, lambda: torch._C._functorch._vmap_decrement_nesting())
        self.proxy = tx.output.create_proxy(
            "call_function",
            torch._functorch.predispatch._vmap_increment_nesting,
            (batch_size.as_proxy(), randomness),
            {},
        )
        return variables.ConstantVariable.create(vmap_level)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function",
            torch._functorch.predispatch._vmap_decrement_nesting,
            (),
            {},
        )
        return variables.ConstantVariable.create(None)


class GradModeVariable(ContextWrappingVariable):
    """represents torch.{no_grad,enable_grad,set_grad_mode}()"""

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.GRAD_MODE)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslator",
        target_value: Any,
        initialized: bool = False,
        **kwargs: Any,
    ) -> "GradModeVariable":
        var = GradModeVariable(
            target_values=[target_value],
            initial_values=[torch.is_grad_enabled()],
            **kwargs,
        )
        if initialized:
            var._call_func(tx, var.target_values)
        return var

    def __init__(
        self,
        target_values: Any,
        initial_values: Optional[Sequence[bool]] = None,
        initialized: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        install_guard(self._guards_singleton)

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        self._call_func(tx, self.target_values)
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self._call_func(tx, self.initial_values)
        return variables.ConstantVariable.create(None)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self._call_func(tx, self.initial_values)  # undo eager initialization
        return super().call_function(tx, args, kwargs)

    def _call_func(self, tx: "InstructionTranslator", values: Any) -> None:
        assert len(values) == 1
        value = values[0]
        # Coalesce grad mode mutations
        if torch.is_grad_enabled() != value:
            tx.output.create_node(
                "call_function", torch._C._set_grad_enabled, (value,), {}
            )
            torch._C._set_grad_enabled(value)

    def module_name(self) -> str:
        return "torch"

    def fn_name(self) -> str:
        return "set_grad_enabled"


class InferenceModeVariable(ContextWrappingVariable):
    @staticmethod
    def create(
        tx: "InstructionTranslator", target_value: Any, **kwargs: Any
    ) -> "InferenceModeVariable":
        var = InferenceModeVariable(
            [target_value], initial_values=torch.is_inference_mode_enabled(), **kwargs
        )
        return var

    def __init__(
        self,
        target_values: Any,
        initial_values: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if initial_values is None:
            # This must be called here since function defaults are evaluated at import time
            initial_values = torch.is_inference_mode_enabled()
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup_assert()
        tx.output.create_node(
            "call_function",
            torch.autograd.grad_mode._exit_inference_mode,
            (self.proxy,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        disabled_inference_mode_forcibly = False
        if (
            torch._dynamo.config.fake_tensor_disable_inference_mode
            and self.target_values[0]
        ):
            # Do not set the inference mode because we keep it off during
            # compilation. Set the grad_enabled to False to reflect the relevant
            # part of inference_mode to torch.compile.
            disabled_inference_mode_forcibly = True
            prior = torch.is_grad_enabled()
            torch._C._set_grad_enabled(False)
        else:
            ctx = torch.autograd.grad_mode._enter_inference_mode(*self.target_values)

        def cleanup_hook() -> None:
            if disabled_inference_mode_forcibly:
                torch._C._set_grad_enabled(prior)
            else:
                torch.autograd.grad_mode._exit_inference_mode(ctx)

        self.set_cleanup_hook(tx, cleanup_hook)
        self.proxy = tx.output.create_node(
            "call_function",
            torch.autograd.grad_mode._enter_inference_mode,
            (*self.target_values,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "torch"

    def fn_name(self) -> str:
        return "inference_mode"


class CUDADeviceVariable(ContextWrappingVariable):
    """represents torch.cuda.device"""

    @staticmethod
    def create(
        tx: "InstructionTranslator", device: Any, **kwargs: Any
    ) -> "CUDADeviceVariable":
        var = CUDADeviceVariable(
            target_values=[torch.cuda._get_device_index(device, optional=True)],
            initial_values=None,
            **kwargs,
        )
        return var

    def __init__(
        self,
        target_values: Any,
        initial_values: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup_assert()
        tx.output.create_node(
            "call_function",
            torch.cuda._maybe_exchange_device,
            (self.proxy,),
            {},
        )
        return variables.ConstantVariable.create(False)

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        prev_idx = torch.cuda._exchange_device(*self.target_values)
        self.set_cleanup_hook(tx, lambda: torch.cuda._maybe_exchange_device(prev_idx))
        self.proxy = tx.output.create_node(
            "call_function",
            torch.cuda._exchange_device,
            (*self.target_values,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "torch.cuda"

    def fn_name(self) -> str:
        return "device"


class TorchFunctionDisableVariable(ContextWrappingVariable):
    """represents whether torch function overrides are enabled or not"""

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.TORCH_FUNCTION_STATE)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslator", **kwargs: Any
    ) -> "TorchFunctionDisableVariable":
        var = TorchFunctionDisableVariable(
            target_values=[],
            initial_values=[],
            **kwargs,
        )
        return var

    def __init__(
        self,
        target_values: Sized,
        initial_values: Optional[Sized] = None,
        only_subclass: bool = True,
        **kwargs: Any,
    ) -> None:
        assert len(target_values) == 0
        assert initial_values is not None and len(initial_values) == 0
        from ..symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()
        self.only_subclass = only_subclass
        self.initial_torch_function_subclass_enabled = (
            tx.symbolic_torch_function_state.torch_function_subclass_enabled
        )
        self.initial_torch_function_mode_enabled = (
            tx.symbolic_torch_function_state.torch_function_mode_enabled
        )

        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        install_guard(self._guards_singleton)

    def set_cleanup_hook(
        self,
        tx: "InstructionTranslator",
        cleanup_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        if cleanup_fn is None:

            def cleanup_fn() -> None:
                tx.symbolic_torch_function_state.torch_function_subclass_enabled = (
                    self.initial_torch_function_subclass_enabled
                )
                if not self.only_subclass:
                    tx.symbolic_torch_function_state.torch_function_mode_enabled = (
                        self.initial_torch_function_subclass_enabled
                    )

        self.cleanup_fn = cleanup_fn
        tx.output.add_cleanup_hook(self.cleanup)

    def _call_func(self, tx: "InstructionTranslator", values: Sized) -> None:
        assert len(values) == 0
        tx.symbolic_torch_function_state.torch_function_subclass_enabled = False
        if not self.only_subclass:
            tx.symbolic_torch_function_state.torch_function_mode_enabled = False

    def module_name(self) -> str:
        return "torch._C"

    def fn_name(self) -> str:
        if self.only_subclass:
            return "DisableTorchFunctionSubclass"
        return "DisableTorchFunction"


class DeterministicAlgorithmsVariable(ContextWrappingVariable):
    """represents torch.{are_deterministic_algorithms_enabled,use_deterministic_algorithms}()"""

    _guards_singleton = Guard(
        GlobalStateSource(),
        GuardBuilder.DETERMINISTIC_ALGORITHMS,  # type: ignore[arg-type]
    )

    @staticmethod
    def create(
        tx: "InstructionTranslator", target_value: bool, **kwargs: Any
    ) -> "DeterministicAlgorithmsVariable":
        var = DeterministicAlgorithmsVariable(
            target_values=[target_value],
            initial_values=[torch.are_deterministic_algorithms_enabled()],
            **kwargs,
        )
        var._call_func(tx, [target_value])
        var.set_cleanup_hook(tx)
        return var

    def __init__(
        self,
        target_values: Sequence[bool],
        initial_values: Optional[Sequence[bool]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        install_guard(self._guards_singleton)

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        return variables.ConstantVariable.create(None)

    def _call_func(self, tx: "InstructionTranslator", values: Sequence[bool]) -> None:
        assert len(values) == 1
        value = values[0]
        tx.output.create_node(
            "call_function", torch._C._set_deterministic_algorithms, (value,), {}
        )
        torch._C._set_deterministic_algorithms(value)

    def module_name(self) -> str:
        return "torch"

    def fn_name(self) -> str:
        return "use_deterministic_algorithms"


class DisabledSavedTensorsHooksVariable(ContextWrappingVariable):
    """represents torch.autograd.graph.disable_saved_tensors_hook."""

    @staticmethod
    def create(
        tx: "InstructionTranslator", target_value: Optional[str], **kwargs: Any
    ) -> "DisabledSavedTensorsHooksVariable":
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

    def __init__(
        self,
        target_values: Sequence[Optional[str]],
        initial_values: Optional[Sequence[Optional[str]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        return variables.ConstantVariable.create(None)

    def _call_func(
        self, tx: "InstructionTranslator", values: Sequence[Optional[str]]
    ) -> None:
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

    def module_name(self) -> str:
        return "torch.autograd.graph"

    def fn_name(self) -> str:
        return "disable_saved_tensors_hooks"


class AutocastModeVariable(ContextWrappingVariable):
    @staticmethod
    def create(
        func: torch.amp.autocast_mode.autocast,
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> "AutocastModeVariable":
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

    def __init__(
        self,
        target_values: Sequence[Any],
        initial_values: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup_assert()
        tx.output.create_node(
            "call_function", torch.amp._exit_autocast, (self.proxy,), {}
        )
        return variables.ConstantVariable.create(None)

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        ctx = torch.amp._enter_autocast(*self.target_values)
        self.set_cleanup_hook(tx, lambda: torch.amp._exit_autocast(ctx))
        self.proxy = tx.output.create_node(
            "call_function", torch.amp._enter_autocast, (*self.target_values,), {}
        )
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "torch.amp.autocast_mode"

    def fn_name(self) -> str:
        return "autocast"


class NullContextVariable(ContextWrappingVariable):
    """
    This class represents Python contextlib.nullcontext.
    """

    def __init__(self, target_values: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(target_values=target_values, **kwargs)

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        none = variables.ConstantVariable.create(None)
        return self.target_values if self.target_values else none

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "contextlib"

    def fn_name(self) -> str:
        return "nullcontext"


class ProfilerContextVariable(ContextWrappingVariable):
    """
    This class represents a set of torch profiler context objects, where Dynamo
    ignores all the side-effects in the __init__, __enter__ and __exit__ methods
    by treating the object mostly as a `contextlib.nullcontext`, except for edge
    cases like the `__enter__` method which returns the object itself rather
    than `None`, per implementation of the torch objects.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(target_values=None, **kwargs)

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        return self

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "contextlib"

    def fn_name(self) -> str:
        return "nullcontext"

    def reconstruct(self, cg: "PyCodegen") -> None:
        unimplemented(
            gb_type="torch.profiler object escaped from compiled region",
            context=str(self),
            explanation="Dynamo doesn't support compiling a region that returns a torch.profiler context manager.",
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )


class PreserveVersionContextVariable(ContextWrappingVariable):
    """
    Wraps torch.autograd._unsafe_preserve_version_counter
    """

    @staticmethod
    def _create_lambda_from_tensors(
        tx: "InstructionTranslator",
        tensors: VariableTracker,
    ) -> "PreserveVersionContextVariable":
        if isinstance(tensors, variables.TensorVariable):
            versions = variables.TupleVariable(
                [x.var_getattr(tx, "_version") for x in [tensors]]
            )
            tensors_tuple = variables.TupleVariable([tensors])
        else:
            assert isinstance(tensors, variables.TupleVariable)
            versions = variables.TupleVariable(
                [x.var_getattr(tx, "_version") for x in tensors.items]
            )
            tensors_tuple = tensors
        return PreserveVersionContextVariable(tensors_tuple, versions)

    @staticmethod
    def constructor(tx: "InstructionTranslator") -> VariableTracker:
        return variables.LambdaVariable(
            lambda tensors: PreserveVersionContextVariable._create_lambda_from_tensors(
                tx, tensors
            )
        )

    def __init__(
        self,
        tensors: VariableTracker,
        prev_versions: VariableTracker,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("target_values", None)
        super().__init__(**kwargs)
        self.tensors = tensors
        self.prev_versions = prev_versions
        # The context manager accepts Union[Tensor, Tuple[Tensor]]
        if isinstance(self.tensors, variables.TensorVariable):
            self.tensors = variables.TupleVariable([self.tensors])
        if isinstance(
            self.prev_versions, (variables.ConstantVariable, variables.SymNodeVariable)
        ):
            self.prev_versions = variables.TupleVariable([self.prev_versions])

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        from ..tensor_version_op import _unsafe_set_version_counter

        return variables.TorchInGraphFunctionVariable(
            _unsafe_set_version_counter
        ).call_function(tx, [self.tensors, self.prev_versions], {})

    def reconstruct(self, codegen: "PyCodegen") -> None:
        unimplemented(
            gb_type="torch.autograd._unsafe_preserve_version_counter escaped from compiled region",
            context=str(self),
            explanation=(
                "Dynamo doesn't support compiling a region that returns "
                "a torch.autograd._unsafe_preserve_version_counter context manager."
            ),
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )


class FSDPParamGroupUseTrainingStateVariable(ContextWrappingVariable):
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FSDP_TRAINING_STATE)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslator",
        param_group_var: Any,
        target_value: Any,
        **kwargs: Any,
    ) -> "FSDPParamGroupUseTrainingStateVariable":
        var = FSDPParamGroupUseTrainingStateVariable(
            param_group_var=param_group_var,
            target_values=[target_value],
            initial_values=[param_group_var.value._training_state],
            **kwargs,
        )
        return var

    def __init__(
        self,
        param_group_var: Any,
        target_values: Sequence[Any],
        initial_values: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.param_group_var = param_group_var
        install_guard(self._guards_singleton)

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        self._call_func(tx, self.target_values)
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self._call_func(tx, self.initial_values)  # type: ignore[arg-type]
        return variables.ConstantVariable.create(None)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # undo eager initialization
        self._call_func(tx, self.initial_values)  # type: ignore[arg-type]
        return super().call_function(tx, args, kwargs)

    def _call_func(self, tx: "InstructionTranslator", values: Sequence[Any]) -> None:
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

    def module_name(self) -> str:
        return "torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup"

    def fn_name(self) -> str:
        return "use_training_state"


class SDPAKernelVariable(ContextWrappingVariable):
    """represents torch.nn.attention.sdpa_kernel"""

    @staticmethod
    def create(
        tx: "InstructionTranslator",
        backends: Any,
        set_priority: bool = False,
        **kwargs: Any,
    ) -> "SDPAKernelVariable":
        if isinstance(backends, torch.nn.attention.SDPBackend):
            backends = [backends]
        var = SDPAKernelVariable(
            target_values=backends,
            initial_values=None,
            set_priority=set_priority,
            **kwargs,
        )
        return var

    def __init__(
        self,
        target_values: list[torch.nn.attention.SDPBackend],
        initial_values: Any = None,
        set_priority: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.set_priority = set_priority

    @staticmethod
    def _backends_to_nodes(
        tx: "InstructionTranslator",
        backends: list[Any],
    ) -> list[Any]:
        # convert to/from string in order to bake the backend into FX graph
        nodes = [
            tx.output.create_node(
                "call_function",
                torch.nn.attention._backend_from_string,
                (backend.name,),
                {},
            )
            for backend in backends
        ]
        return nodes

    def enter(self, tx: "InstructionTranslator") -> VariableTracker:
        self.prev_backends = torch.nn.attention._cur_sdpa_kernel_backends(
            with_priority=self.set_priority
        )
        self.set_cleanup_hook(
            tx,
            lambda: torch.nn.attention._sdpa_kernel(
                self.prev_backends, set_priority=self.set_priority
            ),
        )
        torch.nn.attention._sdpa_kernel(
            self.target_values, set_priority=self.set_priority
        )
        arg = self._backends_to_nodes(tx, self.target_values)
        tx.output.create_node(
            "call_function",
            torch.nn.attention._sdpa_kernel,
            (arg, bool(self.set_priority)),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup_assert()
        arg = self._backends_to_nodes(tx, self.prev_backends)
        tx.output.create_node(
            "call_function",
            torch.nn.attention._sdpa_kernel,
            (arg, bool(self.set_priority)),
            {},
        )
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "torch.nn.attention"

    # use a private version of sdpa_kernel that accepts variadic arguments
    # since dynamo reconstructs the contents of target_values one-by-one
    def fn_name(self) -> str:
        return "_sdpa_kernel_variadic"


class FxTracebackAnnotateVariable(ContextWrappingVariable):
    """
    fx.traceback.annotate is a context manager that allows users to annotate the
    fx graph nodes with custom metadata. In the context of Dynamo, we don't have
    to trace the body of the context manager. Instead we want to directly run
    the body of the context manager, so the Dynamo created Fx graphs have the
    right custom metadata. This variable tracker just runs __enter__ and
    __exit__ method (instead of tracing).
    """

    def __init__(
        self, target_values: Any, initial_values: Any = None, **kwargs: Any
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def enter(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        # Run the annotation ctx manager in eager. Also ensure that
        # preserve_node_meta context manager is setup. This is important to pass
        # on the metadata to the create_proxy nodes.
        stack = ExitStack()
        stack.enter_context(torch.fx.traceback.annotate(self.target_values))
        stack.enter_context(torch.fx.traceback.preserve_node_meta())
        self.set_cleanup_hook(tx, lambda: stack.close())
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "torch.fx.traceback"

    def fn_name(self) -> str:
        return "annotate"

    def reconstruct_type(self, codegen: "PyCodegen") -> None:
        unimplemented(
            gb_type="torch.fx.traceback.annotate escaped from compiled region",
            context=str(self),
            explanation="Dynamo doesn't support graph break on torch.fx.traceback.annotate.",
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )


class DynamoConfigPatchVariable(ContextWrappingVariable):
    """represents torch._dynamo.patch_dynamo_config"""

    # NOTE: no need to guard on dynamo config because dynamo config should not affect soundness
    # (though it may affect tracing behavior)
    def __init__(self, target_values: dict[str, Any], **kwargs: Any) -> None:
        target_values_tuple = tuple(target_values.items())
        super().__init__(
            target_values=(target_values_tuple,), initial_values=None, **kwargs
        )
        initial_values_dict = {}
        for key, _ in target_values_tuple:
            initial_values_dict[key] = torch._dynamo.config.__getattr__(key)  # type: ignore[attr-defined]
        self.initial_values = (tuple(initial_values_dict.items()),)

    def _call_func(self, tx: "InstructionTranslator", values: Any) -> None:
        assert len(values) == 1
        value = values[0]
        # manually patch dynamo config
        for key, val in value:
            torch._dynamo.config.__setattr__(key, val)  # type: ignore[attr-defined]
        # No need to keep track of global side effects because
        # dynamo will properly restore this context manager for
        # unsupported instructions and continuation functions.
        # Dynamo config also should not affect the semantics of the compiled graph.

    def module_name(self) -> str:
        return "torch._dynamo"

    def fn_name(self) -> str:
        return "patch_dynamo_config"


class ErrorOnGraphBreakVariable(ContextWrappingVariable):
    """represents torch._dynamo.error_on_graph_break"""

    def __init__(self, error_on_graph_break: bool, **kwargs: Any) -> None:
        super().__init__(
            target_values=(error_on_graph_break,),
            initial_values=(_get_error_on_graph_break(),),
            **kwargs,
        )

    def _call_func(self, tx: "InstructionTranslator", values: Sequence[bool]) -> None:
        assert len(values) == 1
        _set_error_on_graph_break(values[0])

    def module_name(self) -> str:
        return "torch._dynamo"

    def fn_name(self) -> str:
        return "error_on_graph_break"


class WithEnterFunctionVariable(VariableTracker):
    def __init__(
        self,
        ctx: Union[ContextWrappingVariable, GenericContextWrappingVariable],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ctx = ctx

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        assert not args
        assert not kwargs
        # NOTE: we assume that the instruction immediately after the current CALL instruction
        # is the first instruction of the block.
        # pyrefly: ignore [bad-argument-type]
        return tx.enter_ctx(self.ctx, tx.current_instruction)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        try:
            type_str = f"{self.ctx.module_name()}.{self.ctx.fn_name()}"
        except NotImplementedError:
            type_str = str(type(self.ctx))
        unimplemented(
            gb_type="Attempted to reconstruct context manager's __enter__ method",
            context=str(self.ctx),
            explanation=f"Attempted to reconstruct context manager {type_str} while tracing `with ...:`",
            hints=[
                "It is likely there is a graph break while tracing `with ctx:` "
                "but outside the actual `ctx.__enter__()` method. "
                "`torch.compile` does not expect this to happen.",
                *graph_break_hints.DIFFICULT,
                *graph_break_hints.DYNAMO_BUG,
            ],
        )


class WithExitFunctionVariable(VariableTracker):
    _nonvar_fields = {
        "target",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        ctx: Union[ContextWrappingVariable, GenericContextWrappingVariable],
        target: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(
            ctx, (ContextWrappingVariable, GenericContextWrappingVariable)
        )
        self.ctx = ctx
        self.target = target

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        assert not kwargs
        return self.ctx.exit(tx, *args)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Note here we reconstruct the context manager rather than the
        # exit function.  The handler generated by BlockStackEntry
        # will re-enter the context in the resume function.
        self.ctx.reconstruct_type(codegen)  # type: ignore[union-attr]
        if codegen.tx.output.partial_convert:
            if sys.version_info >= (3, 11):
                codegen.append_output(create_instruction("PUSH_NULL"))
                if sys.version_info < (3, 13):
                    codegen.append_output(create_instruction("SWAP", arg=2))
            # We rely on classes subtyping `GenericContextWrappingVariable`
            # to implement these fns and have these attributes
            codegen.extend_output(
                [codegen.create_load_const(val) for val in self.ctx.target_values]  # type: ignore[union-attr]
            )
            codegen.extend_output(
                create_call_function(len(self.ctx.target_values), False)  # type: ignore[union-attr]
            )
            codegen.append_output(create_setup_with(self.target))
            codegen.append_output(create_instruction("POP_TOP"))
