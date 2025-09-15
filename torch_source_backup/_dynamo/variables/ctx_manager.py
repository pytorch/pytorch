# mypy: ignore-errors

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
from typing import TYPE_CHECKING, Union

import torch._C
from torch._guards import Guard

from .. import graph_break_hints, variables
from ..bytecode_transformation import (
    create_call_function,
    create_instruction,
    create_setup_with,
)
from ..device_interface import get_interface_for_device
from ..exc import unimplemented_v2
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalStateSource
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

    def __init__(self, target_values, initial_values=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.target_values = target_values
        self.initial_values = initial_values

    def enter(self, tx):
        self._call_func(tx, self.target_values)
        self.set_cleanup_hook(tx)
        return variables.ConstantVariable.create(None)

    def set_cleanup_hook(self, tx: "InstructionTranslator", fn=None):
        if fn is None:

            def fn():
                self._call_func(tx, self.initial_values)

        self.cleanup_fn = fn
        tx.output.add_cleanup_hook(self.cleanup)

    def exit(self, tx: "InstructionTranslator", *args):
        self.cleanup_assert()
        return variables.ConstantVariable.create(None)

    def reconstruct_type(self, codegen: "PyCodegen"):
        codegen(
            AttrSource(codegen.tx.import_source(self.module_name()), self.fn_name())
        )

    def reconstruct(self, codegen: "PyCodegen"):
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
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
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

        if isinstance(args[0], SkipFunctionVariable):
            return WrappedSkipFunctionVariable(args[0], self)

        if isinstance(args[0], UserMethodVariable):
            return WrappedUserMethodVariable(args[0], self)

        if isinstance(args[0], UserFunctionVariable):
            return WrappedUserFunctionVariable(args[0], self)

    def supports_graph_breaks(self):
        return True

    def exit_on_graph_break(self):
        return True

    def cleanup(self):
        if self.cleanup_fn is not None:
            self.cleanup_fn()
            self.cleanup_fn = None

    def cleanup_assert(self):
        assert self.cleanup_fn, "multiple exits?"
        self.cleanup()


class GenericContextWrappingVariable(UserDefinedObjectVariable):
    # Some methods in ContextWrappingVariable assumes the arguments are
    # python constants. Which might not always be the case here.
    def __init__(self, cm_obj, **kwargs) -> None:
        assert cm_obj is not None
        super().__init__(
            value=cm_obj,
            value_type=cm_obj.__class__,
            **kwargs,
        )
        self.cm_obj = cm_obj

    def module_name(self):
        return self.cm_obj.__module__

    def fn_name(self):
        return type(self.cm_obj).__name__

    def enter(self, tx):
        source = None if self.source is None else AttrSource(self.source, "__enter__")
        return variables.UserMethodVariable(
            self.cm_obj.__enter__.__func__,
            self,
            source=source,
        ).call_function(tx, [], {})

    def exit(self, tx: "InstructionTranslator", *args):
        source = None if self.source is None else AttrSource(self.source, "__exit__")
        x = variables.UserMethodVariable(
            self.cm_obj.__exit__.__func__,
            self,
            source=source,
        ).call_function(tx, args, {})
        tx.active_generic_context_managers.pop()
        return x

    def supports_graph_breaks(self):
        return False

    def exit_on_graph_break(self):
        return True


class GradInplaceRequiresGradCtxManagerVariable(ContextWrappingVariable):
    """represents torch grad requires grad"""

    @staticmethod
    def create(tx: "InstructionTranslator", target_values, **kwargs):
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
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch.set_inplace_requires_grad_allowed,
            (enabled,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(self, tx: "InstructionTranslator", *args):
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
    def create(tx: "InstructionTranslator", target_values, **kwargs):
        return TemporarilyPopInterpreterStackCtxManagerVariable(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx):
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

    def exit(self, tx: "InstructionTranslator", *args):
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
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FUNCTORCH_STACK_MATCH)

    @staticmethod
    def create(tx: "InstructionTranslator", **kwargs):
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
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch._jvp_increment_nesting,
            (),
            {},
        )
        return variables.ConstantVariable.create(jvp_level)

    def exit(self, tx: "InstructionTranslator", *args):
        self.cleanup()
        tx.output.create_node(
            "call_function", torch._C._functorch._jvp_decrement_nesting, (), {}
        )
        return variables.ConstantVariable.create(None)


class SetFwdGradEnabledContextManager(ContextWrappingVariable):
    """represents torch.autograd.forward_ad._set_fwd_grad_enabled() to enable/disable fwd grad"""

    @staticmethod
    def create(tx: "InstructionTranslator", target_values, **kwargs):
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
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._set_fwd_grad_enabled,
            (mode,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(self, tx: "InstructionTranslator", *args):
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

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.DUAL_LEVEL)

    @staticmethod
    def create(tx: "InstructionTranslator", **kwargs):
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
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._enter_dual_level,
            (),
            {},
        )
        return variables.ConstantVariable.create(self.new_level)

    def exit(self, tx: "InstructionTranslator", *args):
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
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FUNCTORCH_STACK_MATCH)

    @staticmethod
    def create(tx: "InstructionTranslator", **kwargs):
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
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch._grad_increment_nesting,
            (),
            {},
        )
        return variables.ConstantVariable.create(grad_level)

    def exit(self, tx: "InstructionTranslator", *args):
        self.cleanup()
        tx.output.create_node(
            "call_function", torch._C._functorch._grad_decrement_nesting, (), {}
        )
        return variables.ConstantVariable.create(None)


class CatchWarningsCtxManagerVariable(ContextWrappingVariable):
    """Delay a call to warnings.catch_warnings"""

    @staticmethod
    def create(tx: "InstructionTranslator", catch_warnings_args):
        return CatchWarningsCtxManagerVariable(
            catch_warnings_args=catch_warnings_args,
            target_values=None,
            initial_values=None,
        )

    def __init__(self, catch_warnings_args, **kwargs) -> None:
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
    def create(tx: "InstructionTranslator", target_values, **kwargs):
        var = VmapIncrementNestingCtxManagerVariable(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )
        return var

    def enter(self, tx):
        install_guard(self._guards_singleton)
        batch_size, randomness = self.target_values
        if isinstance(batch_size, variables.SymNodeVariable):
            batch_size_value = batch_size.sym_num
            batch_size_node = batch_size.as_proxy().node
        else:
            batch_size_value = batch_size.as_python_constant()
            batch_size_node = batch_size.as_python_constant()
        randomness = randomness.as_python_constant()
        vmap_level = torch._C._functorch._vmap_increment_nesting(
            batch_size_value, randomness
        )
        self.set_cleanup_hook(tx, lambda: torch._C._functorch._vmap_decrement_nesting())
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch._vmap_increment_nesting,
            (batch_size_node, randomness),
            {},
        )
        return variables.ConstantVariable.create(vmap_level)

    def exit(self, tx: "InstructionTranslator", *args):
        self.cleanup()
        tx.output.create_node(
            "call_function", torch._C._functorch._vmap_decrement_nesting, (), {}
        )
        return variables.ConstantVariable.create(None)


class GradModeVariable(ContextWrappingVariable):
    """represents torch.{no_grad,enable_grad,set_grad_mode}()"""

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.GRAD_MODE)

    @staticmethod
    def create(tx: "InstructionTranslator", target_value, initialized=False, **kwargs):
        var = GradModeVariable(
            target_values=[target_value],
            initial_values=[torch.is_grad_enabled()],
            **kwargs,
        )
        if initialized:
            var._call_func(tx, var.target_values)
        return var

    def __init__(
        self, target_values, initial_values=None, initialized=True, **kwargs
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        install_guard(self._guards_singleton)

    def enter(self, tx):
        self._call_func(tx, self.target_values)
        return variables.ConstantVariable.create(None)

    def exit(self, tx: "InstructionTranslator", *args):
        self._call_func(tx, self.initial_values)
        return variables.ConstantVariable.create(None)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ):
        self._call_func(tx, self.initial_values)  # undo eager initialization
        return super().call_function(tx, args, kwargs)

    def _call_func(self, tx: "InstructionTranslator", values):
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
    def create(tx: "InstructionTranslator", target_value, **kwargs):
        var = InferenceModeVariable(
            [target_value], initial_values=torch.is_inference_mode_enabled(), **kwargs
        )
        return var

    def __init__(
        self,
        target_values,
        initial_values=None,
        **kwargs,
    ) -> None:
        if initial_values is None:
            # This must be called here since function defaults are evaluated at import time
            initial_values = torch.is_inference_mode_enabled()
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.target_values = target_values

    def exit(self, tx: "InstructionTranslator", *args):
        self.cleanup_assert()
        tx.output.create_node(
            "call_function",
            torch.autograd.grad_mode._exit_inference_mode,
            (self.proxy,),
            {},
        )

    def enter(self, tx):
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

        def cleanup_hook():
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

    def module_name(self):
        return "torch"

    def fn_name(self):
        return "inference_mode"


class CUDADeviceVariable(ContextWrappingVariable):
    """represents torch.cuda.device"""

    @staticmethod
    def create(tx: "InstructionTranslator", device, **kwargs):
        var = CUDADeviceVariable(
            target_values=[torch.cuda._get_device_index(device, optional=True)],
            initial_values=None,
            **kwargs,
        )
        return var

    def __init__(
        self,
        target_values,
        initial_values=None,
        **kwargs,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.target_values = target_values

    def exit(self, tx: "InstructionTranslator", *args):
        self.cleanup_assert()
        tx.output.create_node(
            "call_function",
            torch.cuda._maybe_exchange_device,
            (self.proxy,),
            {},
        )
        return variables.ConstantVariable.create(False)

    def enter(self, tx):
        prev_idx = torch.cuda._exchange_device(*self.target_values)
        self.set_cleanup_hook(tx, lambda: torch.cuda._maybe_exchange_device(prev_idx))
        self.proxy = tx.output.create_node(
            "call_function",
            torch.cuda._exchange_device,
            (*self.target_values,),
            {},
        )

    def module_name(self):
        return "torch.cuda"

    def fn_name(self):
        return "device"


class TorchFunctionDisableVariable(ContextWrappingVariable):
    """represents whether torch function overrides are enabled or not"""

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.TORCH_FUNCTION_STATE)

    @staticmethod
    def create(tx: "InstructionTranslator", **kwargs):
        var = TorchFunctionDisableVariable(
            target_values=[],
            initial_values=[],
            **kwargs,
        )
        return var

    def __init__(
        self, target_values, initial_values=None, only_subclass=True, **kwargs
    ) -> None:
        assert len(target_values) == 0
        assert len(initial_values) == 0
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

    def set_cleanup_hook(self, tx: "InstructionTranslator", fn=None):
        if fn is None:

            def fn():
                tx.symbolic_torch_function_state.torch_function_subclass_enabled = (
                    self.initial_torch_function_subclass_enabled
                )
                if not self.only_subclass:
                    tx.symbolic_torch_function_state.torch_function_mode_enabled = (
                        self.initial_torch_function_subclass_enabled
                    )

        self.cleanup_fn = fn
        tx.output.add_cleanup_hook(self.cleanup)

    def _call_func(self, tx: "InstructionTranslator", values):
        assert len(values) == 0
        tx.symbolic_torch_function_state.torch_function_subclass_enabled = False
        if not self.only_subclass:
            tx.symbolic_torch_function_state.torch_function_mode_enabled = False

    def module_name(self):
        return "torch._C"

    def fn_name(self):
        if self.only_subclass:
            return "DisableTorchFunctionSubclass"
        return "DisableTorchFunction"


class DeterministicAlgorithmsVariable(ContextWrappingVariable):
    """represents torch.{are_deterministic_algorithms_enabled,use_deterministic_algorithms}()"""

    _guards_singleton = Guard(
        GlobalStateSource(), GuardBuilder.DETERMINISTIC_ALGORITHMS
    )

    @staticmethod
    def create(tx: "InstructionTranslator", target_value, **kwargs):
        var = DeterministicAlgorithmsVariable(
            target_values=[target_value],
            initial_values=[torch.are_deterministic_algorithms_enabled()],
            **kwargs,
        )
        var._call_func(tx, [target_value])
        var.set_cleanup_hook(tx)
        return var

    def __init__(self, target_values, initial_values=None, **kwargs) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        install_guard(self._guards_singleton)

    def enter(self, tx):
        return variables.ConstantVariable.create(None)

    def _call_func(self, tx: "InstructionTranslator", values):
        assert len(values) == 1
        value = values[0]
        (
            tx.output.create_node(
                "call_function", torch._C._set_deterministic_algorithms, (value,), {}
            ),
        )
        torch._C._set_deterministic_algorithms(value)

    def module_name(self):
        return "torch"

    def fn_name(self):
        return "use_deterministic_algorithms"


class DisabledSavedTensorsHooksVariable(ContextWrappingVariable):
    """represents torch.autograd.graph.disable_saved_tensors_hook."""

    @staticmethod
    def create(tx: "InstructionTranslator", target_value, **kwargs):
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

    def __init__(self, target_values, initial_values=None, **kwargs) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def enter(self, tx):
        return variables.ConstantVariable.create(None)

    def _call_func(self, tx: "InstructionTranslator", values):
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

    def __init__(self, target_values, initial_values=None, **kwargs) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.target_values = target_values

    def exit(self, tx: "InstructionTranslator", *args):
        self.cleanup_assert()
        tx.output.create_node(
            "call_function", torch.amp._exit_autocast, (self.proxy,), {}
        )
        return variables.ConstantVariable.create(None)

    def enter(self, tx):
        ctx = torch.amp._enter_autocast(*self.target_values)
        self.set_cleanup_hook(tx, lambda: torch.amp._exit_autocast(ctx))
        self.proxy = tx.output.create_node(
            "call_function", torch.amp._enter_autocast, (*self.target_values,), {}
        )

    def module_name(self):
        return "torch.amp.autocast_mode"

    def fn_name(self):
        return "autocast"


class NullContextVariable(ContextWrappingVariable):
    """
    This class represents Python contextlib.nullcontext.
    """

    def __init__(self, target_values=None, **kwargs) -> None:
        super().__init__(target_values=target_values, **kwargs)

    def enter(self, tx):
        return variables.ConstantVariable.create(None)

    def exit(self, tx: "InstructionTranslator", *args):
        return variables.ConstantVariable.create(None)

    def module_name(self):
        return "contextlib"

    def fn_name(self):
        return "nullcontext"


class ProfilerContextVariable(ContextWrappingVariable):
    """
    This class represents a set of torch profiler context objects, where Dynamo
    ignores all the side-effects in the __init__, __enter__ and __exit__ methods
    by treating the object mostly as a `contextlib.nullcontext`, except for edge
    cases like the `__enter__` method which returns the object itself rather
    than `None`, per implementation of the torch objects.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(target_values=None, **kwargs)

    def enter(self, tx):
        return self

    def exit(self, tx: "InstructionTranslator", *args):
        return variables.ConstantVariable.create(None)

    def module_name(self):
        return "contextlib"

    def fn_name(self):
        return "nullcontext"

    def reconstruct(self, cg):
        unimplemented_v2(
            gb_type="torch.profiler object escaped from compiled region",
            context=str(self),
            explanation="Dynamo doesn't support compiling a region that returns a torch.profiler context manager.",
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )


class StreamContextVariable(ContextWrappingVariable):
    @staticmethod
    def create(tx: "InstructionTranslator", target_value, **kwargs):
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

    def __init__(self, target_values, device, initial_values=None, **kwargs) -> None:
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

    def exit(self, tx: "InstructionTranslator", *args):
        tx.output.create_proxy(
            "call_function",
            self.set_stream,
            (self.initial_values[0].as_proxy(),),
            {},
        )
        self.cleanup_assert()


class PreserveVersionContextVariable(ContextWrappingVariable):
    """
    Wraps torch.autograd._unsafe_preserve_version_counter
    """

    @staticmethod
    def _create_lambda_from_tensors(tx, tensors):
        if isinstance(tensors, variables.TensorVariable):
            versions = variables.TupleVariable(
                [x.var_getattr(tx, "_version") for x in [tensors]]
            )
            tensors = variables.TupleVariable([tensors])
        else:
            versions = variables.TupleVariable(
                [x.var_getattr(tx, "_version") for x in tensors.items]
            )
        return PreserveVersionContextVariable(tensors, versions)

    @staticmethod
    def constructor(tx):
        return variables.LambdaVariable(
            lambda tensors: PreserveVersionContextVariable._create_lambda_from_tensors(
                tx, tensors
            )
        )

    def __init__(self, tensors, prev_versions, **kwargs) -> None:
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

    def enter(self, tx):
        pass

    def exit(self, tx: "InstructionTranslator", *args):
        from ..tensor_version_op import _unsafe_set_version_counter

        return variables.TorchInGraphFunctionVariable(
            _unsafe_set_version_counter
        ).call_function(tx, [self.tensors, self.prev_versions], {})

    def reconstruct(self, codegen: "PyCodegen"):
        unimplemented_v2(
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
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FSDP_TRAINING_STATE)

    @staticmethod
    def create(tx: "InstructionTranslator", param_group_var, target_value, **kwargs):
        var = FSDPParamGroupUseTrainingStateVariable(
            param_group_var=param_group_var,
            target_values=[target_value],
            initial_values=[param_group_var.value._training_state],
            **kwargs,
        )
        return var

    def __init__(
        self, param_group_var, target_values, initial_values=None, **kwargs
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.param_group_var = param_group_var
        install_guard(self._guards_singleton)

    def enter(self, tx):
        self._call_func(tx, self.target_values)
        return variables.ConstantVariable.create(None)

    def exit(self, tx: "InstructionTranslator", *args):
        self._call_func(tx, self.initial_values)
        return variables.ConstantVariable.create(None)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ):
        self._call_func(tx, self.initial_values)  # undo eager initialization
        return super().call_function(tx, args, kwargs)

    def _call_func(self, tx: "InstructionTranslator", values):
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
        return "torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup"

    def fn_name(self):
        return "use_training_state"


class SDPAKernelVariable(ContextWrappingVariable):
    """represents torch.nn.attention.sdpa_kernel"""

    @staticmethod
    def create(tx: "InstructionTranslator", backends, set_priority=False, **kwargs):
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
        initial_values=None,
        set_priority: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.set_priority = set_priority

    @staticmethod
    def _backends_to_nodes(tx, backends):
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

    def enter(self, tx):
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

    def exit(self, tx: "InstructionTranslator", *args):
        self.cleanup_assert()
        arg = self._backends_to_nodes(tx, self.prev_backends)
        tx.output.create_node(
            "call_function",
            torch.nn.attention._sdpa_kernel,
            (arg, bool(self.set_priority)),
            {},
        )
        return variables.ConstantVariable.create(None)

    def module_name(self):
        return "torch.nn.attention"

    # use a private version of sdpa_kernel that accepts variadic arguments
    # since dynamo reconstructs the contents of target_values one-by-one
    def fn_name(self):
        return "_sdpa_kernel_variadic"


class StreamVariable(VariableTracker):
    def __init__(self, proxy, value, device, **kwargs) -> None:
        if proxy is not None and "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == value
        assert value.device.type == device.type, (
            "stream value is not equal to the passed device"
        )
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.device = device

    def python_type(self):
        return torch.Stream

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        assert hasattr(self.value, name), f"no stream method found named {name}"

        from ..utils import cmp_name_to_op_mapping, proxy_args_kwargs
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
        elif name in cmp_name_to_op_mapping and len(args) == 1 and not kwargs:
            # NB : Checking for mutation is necessary because we compare
            # constant values
            other = args[0]
            if not isinstance(other, StreamVariable):
                return variables.ConstantVariable.create(NotImplemented)
            return variables.ConstantVariable.create(
                cmp_name_to_op_mapping[name](self.value, other.value)
            )

        return super().call_method(tx, name, args, kwargs)

    def as_proxy(self):
        return self.proxy

    def reconstruct(self, codegen: "PyCodegen"):
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
    def __init__(self, proxy, value, **kwargs) -> None:
        if proxy is not None and "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == value
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
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
            method_name = (
                f"{type(self.value).__module__}.{type(self.value).__qualname__}.{name}"
            )
            unimplemented_v2(
                gb_type="Unsupported event method",
                context=str(name),
                explanation=f"Dynamo doesn't support tracing the {method_name} method. "
                f"We currently support wait, record, synchronize, and query.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

    def as_proxy(self):
        return self.proxy

    def reconstruct(self, codegen: "PyCodegen"):
        # If we got here, this event is fully subsumed by the graph - this means it is
        # not an input or global
        assert not self.source
        # Similar to stream handling, we lift the event into a global and then codegen bytecode to load it from there.
        prefix = "_event"
        name = codegen.tx.output.install_global_by_id(prefix, self.value)
        codegen.append_output(codegen.create_load_global(name, add=True))


class DynamoConfigPatchVariable(ContextWrappingVariable):
    """represents torch._dynamo.patch_dynamo_config"""

    # NOTE: no need to guard on dynamo config because dynamo config should not affect soundness
    # (though it may affect tracing behavior)
    def __init__(self, target_values, **kwargs) -> None:
        target_values = tuple(target_values.items())
        super().__init__(target_values=(target_values,), initial_values=None, **kwargs)
        self.initial_values = {}
        for key, _ in target_values:
            self.initial_values[key] = torch._dynamo.config.__getattr__(key)
        self.initial_values = (tuple(self.initial_values.items()),)

    def enter(self, tx):
        # resets all config patches at the end of tracing
        self.set_cleanup_hook(tx)
        self._call_func(tx, self.target_values)
        return variables.ConstantVariable.create(None)

    def exit(self, tx: "InstructionTranslator", *args):
        self._call_func(tx, self.initial_values)
        return variables.ConstantVariable.create(None)

    def _call_func(self, tx: "InstructionTranslator", values):
        assert len(values) == 1
        value = values[0]
        # manually patch dynamo config
        for key, val in value:
            torch._dynamo.config.__setattr__(key, val)
        # No need to keep track of global side effects because
        # dynamo will properly restore this context manager for
        # unsupported instructions and continuation functions.
        # Dynamo config also should not affect the semantics of the compiled graph.

    def module_name(self):
        return "torch._dynamo"

    def fn_name(self):
        return "patch_dynamo_config"


class WithExitFunctionVariable(VariableTracker):
    _nonvar_fields = {
        "target",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        ctx: Union[ContextWrappingVariable, GenericContextWrappingVariable],
        target,
        **kwargs,
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
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        assert not kwargs
        return self.ctx.exit(tx, *args)

    def reconstruct(self, codegen: "PyCodegen"):
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
