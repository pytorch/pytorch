# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
import inspect
import logging
import math
import re
from typing import Dict, List, TYPE_CHECKING

import torch._C
import torch._refs
import torch.fx
import torch.nn
from torch._guards import TracingContext
from torch._logging import warning_once
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type

from .. import config, polyfills, variables
from ..codegen import PyCodegen
from ..create_parameter_op import (
    can_convert_to_tracable_parameter,
    new_parameter_placeholder,
    tracable_create_parameter,
)
from ..device_interface import get_registered_device_interfaces
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import CallFunctionNoArgsSource, SyntheticLocalSource
from ..utils import (
    check_unspec_or_constant_args,
    guard_if_dyn,
    has_torch_function,
    hashable,
    product,
    proxy_args_kwargs,
    unwrap_if_wrapper,
)
from .base import VariableTracker
from .ctx_manager import (
    AutocastModeVariable,
    NullContextVariable,
    TorchFunctionDisableVariable,
)
from .distributed import DistributedVariable, ProcessGroupVariable
from .lists import ListVariable, TupleVariable
from .torch_function import (
    can_dispatch_torch_function,
    dispatch_torch_function,
    TorchFunctionModeStackVariable,
)


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    from torch.distributed._composable.fsdp import _fsdp_param_group
except ModuleNotFoundError:
    _fsdp_param_group = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


log = logging.getLogger(__name__)

supported_ctx_manager_classes = dict.fromkeys(
    [
        torch.profiler.profiler.profile,
        torch.autograd.forward_ad._set_fwd_grad_enabled,
        torch.autograd.forward_ad.dual_level,
        torch.autograd.profiler.profile,
        torch.autograd.profiler.record_function,
        torch._C.DisableTorchFunctionSubclass,
        torch._functorch.vmap.vmap_increment_nesting,
        torch._functorch.eager_transforms.grad_increment_nesting,
        torch._functorch.eager_transforms.jvp_increment_nesting,
        torch._functorch.eager_transforms.enable_inplace_requires_grad,
        torch.amp.autocast_mode.autocast,
        torch.autograd.grad_mode.enable_grad,
        torch.autograd.grad_mode.inference_mode,
        torch.autograd.grad_mode.no_grad,
        torch.autograd.grad_mode.set_grad_enabled,
        torch.autograd.graph.disable_saved_tensors_hooks,
        torch.cpu.amp.autocast_mode.autocast,
        torch.cuda.amp.autocast_mode.autocast,
        torch.nn.attention.sdpa_kernel,
        torch.nn.attention._sdpa_kernel_variadic,
    ]
)


REWRITE_OPS_TO_TENSOR_SIZE_METHOD = dict.fromkeys(
    [
        torch._shape_as_tensor,
    ]
)

constant_fold_functions_need_guards = [
    torch.cuda.current_device,
]

constant_fold_functions = [
    torch._assert,
    torch._utils._get_device_index,
    torch._C._get_cublas_allow_tf32,
    torch._C._is_any_autocast_enabled,
    torch.cuda.get_device_properties,
    torch.cuda.is_available,
    torch.distributed.is_available,
    torch.get_autocast_dtype,
    torch.get_autocast_gpu_dtype,
    torch.get_default_dtype,
    torch.is_autocast_cache_enabled,
    torch.is_autocast_cpu_enabled,
    torch.is_autocast_enabled,
    torch.is_complex,
    torch.is_floating_point,
    torch.nn.functional._Reduction.get_enum,  # type: ignore[attr-defined]
    torch.promote_types,
    torch._C._get_privateuse1_backend_name,
    torch.autograd._is_checkpoint_valid,
] + constant_fold_functions_need_guards
if torch.distributed.is_available():
    constant_fold_functions.extend(
        [
            torch.distributed.is_initialized,
            torch.distributed.get_rank,
            torch.distributed.get_world_size,
        ]
    )
# Convert to dict for O(1) access times
constant_fold_functions_need_guards = dict.fromkeys(constant_fold_functions_need_guards)
constant_fold_functions = dict.fromkeys(constant_fold_functions)


tracing_state_functions = {
    torch.jit.is_scripting: False,
    torch.jit.is_tracing: False,
    torch._C._get_tracing_state: None,
    torch.fx._symbolic_trace.is_fx_tracing: False,
    torch.onnx.is_in_onnx_export: False,
    torch._dynamo.external_utils.is_compiling: True,
    torch._utils.is_compiling: True,
    torch.compiler.is_compiling: True,
    torch.compiler.is_dynamo_compiling: True,
    torch.nn.modules.activation._is_make_fx_tracing: False,
}

bin_ops = dict.fromkeys(["add", "sub", "mul", "div", "sqrt"])


@functools.lru_cache(None)
def get_overridable_functions():
    from itertools import chain

    from torch.overrides import get_overridable_functions as get_overridable_functions_

    funcs = set(chain(*get_overridable_functions_().values()))
    more = {
        torch.ones,
        torch.ones_like,
        torch.zeros,
        torch.zeros_like,
        torch.empty,
        torch.full,
    }
    funcs.update(more)
    return funcs


class BaseTorchVariable(VariableTracker):
    """common base for all torch.* functions, classes, modules and other things"""

    @classmethod
    def create_with_source(cls, value, source):
        install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        return cls(value, source=source)

    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    def reconstruct(self, codegen):
        try:
            name = f"{self.value.__module__}.{self.value.__name__}"
        except Exception:
            name = f"torch_obj_{id(self.value)}"
        unique_var_name = "__" + re.sub(r"[^a-zA-Z0-9_]+", "_", name)
        codegen.extend_output(
            codegen.setup_globally_cached(unique_var_name, self.value)
        )

    def as_proxy(self):
        return self.value

    def as_python_constant(self):
        return self.value

    def call_hasattr(self, tx: "InstructionTranslator", name):
        result = hasattr(self.value, name)
        return variables.ConstantVariable.create(result)

    def can_constant_fold_through(self):
        if self.value in constant_fold_functions:
            return True
        return getattr(self.value, "__module__", None) == "math"


class TorchCtxManagerClassVariable(BaseTorchVariable):
    """Points to a context manager class in torch.* that dynamo has implementations"""

    def __repr__(self) -> str:
        return f"TorchCtxManagerClassVariable({self.value})"

    @staticmethod
    def is_matching_cls(value):
        # Unwrap if it's a functools.lru_cache wrapper
        value = unwrap_if_wrapper(value)
        # We can't do isinstance(value, type) check because some ctx managers
        # are implemented as a function decorated by contextlib.contextmanager,
        # E.g., torch._functorch.vmap.vmap_increment_nesting.
        return (
            # Context manager type or function with @contextmanager is callable
            callable(value)
            and (
                hashable(value)  # accesses value.__hash__()
                and value in supported_ctx_manager_classes
            )
        )

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import (
            DisabledSavedTensorsHooksVariable,
            DualLevelContextManager,
            FSDPParamGroupUseTrainingStateVariable,
            GradIncrementNestingCtxManagerVariable,
            GradInplaceRequiresGradCtxManagerVariable,
            GradModeVariable,
            InferenceModeVariable,
            JvpIncrementNestingCtxManagerVariable,
            SDPAKernelVariable,
            SetFwdGradEnabledContextManager,
            StreamVariable,
            VmapIncrementNestingCtxManagerVariable,
        )

        if self.value is torch.no_grad:
            if len(args) == 1 and isinstance(
                args[0], variables.functions.BaseUserFunctionVariable
            ):
                ctx = GradModeVariable.create(tx, False)
                return ctx.call_function(tx, args, kwargs)
            else:
                return GradModeVariable.create(tx, False)
        elif self.value is torch.enable_grad:
            if len(args) == 1 and isinstance(
                args[0], variables.functions.BaseUserFunctionVariable
            ):
                ctx = GradModeVariable.create(tx, True)
                return ctx.call_function(tx, args, kwargs)
            return GradModeVariable.create(tx, True)
        elif self.value is torch.set_grad_enabled and len(args) == 1:
            return GradModeVariable.create(
                tx, args[0].as_python_constant(), initialized=True
            )
        elif self.value is torch.inference_mode:
            assert len(args) <= 1 and len(kwargs) == 0
            inf_mode = args[0].as_python_constant() if len(args) == 1 else True
            return InferenceModeVariable.create(tx, inf_mode)
        elif inspect.isclass(self.value) and issubclass(self.value, torch.Stream):
            from torch._dynamo.variables.builder import wrap_fx_proxy_cls

            return wrap_fx_proxy_cls(
                StreamVariable,
                tx,
                tx.output.create_proxy(
                    "call_function",
                    self.value,
                    (),
                    {},
                ),
            )
        elif self.value in (
            torch.amp.autocast_mode.autocast,
            torch.cuda.amp.autocast,
            torch.cpu.amp.autocast,
        ):
            return AutocastModeVariable.create(self.value, args, kwargs)
        elif self.value in (
            torch.profiler.profile,
            torch.profiler.record_function,
            torch.autograd.profiler.profile,
            torch.autograd.profiler.record_function,
        ):
            warning_once(log, "Profiler function %s will be ignored", self.value)
            return NullContextVariable()
        elif self.value is torch._C.DisableTorchFunctionSubclass:
            assert not (args or kwargs)
            return TorchFunctionDisableVariable.create(tx)
        elif self.value is torch._functorch.vmap.vmap_increment_nesting:
            assert len(args) == 2
            return VmapIncrementNestingCtxManagerVariable.create(
                tx,
                args,
            )
        elif self.value is torch._functorch.eager_transforms.jvp_increment_nesting:
            assert len(args) == 0
            return JvpIncrementNestingCtxManagerVariable.create(tx)
        elif self.value is torch.autograd.forward_ad._set_fwd_grad_enabled:
            assert len(args) == 1
            return SetFwdGradEnabledContextManager.create(
                tx,
                [guard_if_dyn(x) for x in args],
            )
        elif self.value is torch.autograd.forward_ad.dual_level:
            assert len(args) == 0
            return DualLevelContextManager.create(tx)
        elif self.value is torch._functorch.eager_transforms.grad_increment_nesting:
            assert len(args) == 0
            return GradIncrementNestingCtxManagerVariable.create(tx)
        elif (
            self.value is torch._functorch.eager_transforms.enable_inplace_requires_grad
        ):
            assert len(args) == 1
            return GradInplaceRequiresGradCtxManagerVariable.create(
                tx,
                [guard_if_dyn(x) for x in args],
            )
        elif self.value is torch.autograd.graph.disable_saved_tensors_hooks:
            assert len(args) == 1
            return DisabledSavedTensorsHooksVariable.create(
                tx, args[0].as_python_constant()
            )
        elif (
            _fsdp_param_group is not None
            and self.value is _fsdp_param_group.FSDPParamGroup.use_training_state
        ):
            assert len(args) == 2
            return FSDPParamGroupUseTrainingStateVariable.create(
                tx, args[0], args[1].as_python_constant()
            )
        elif self.value is torch.nn.attention.sdpa_kernel:
            assert len(args) == 1 or (len(kwargs) == 1 and "backends" in kwargs)
            backends = args[0] if len(args) == 1 else kwargs["backends"]
            return SDPAKernelVariable.create(tx, backends.as_python_constant())
        elif self.value is torch.nn.attention._sdpa_kernel_variadic:
            return SDPAKernelVariable.create(
                tx, [arg.as_python_constant() for arg in args]
            )

        return super().call_function(tx, args, kwargs)


class TorchInGraphFunctionVariable(BaseTorchVariable):
    """Points to a torch function/method that should be put in FX graph"""

    def __repr__(self) -> str:
        return f"TorchInGraphFunctionVariable({self.value})"

    def get_function(self):
        return self.value

    @staticmethod
    @functools.lru_cache(None)
    def _get_handlers():
        """Build a dict from function -> method to handle it so that we are O(1)
        in terms of the number of function with special handling."""
        handlers = {}

        def register(*fns):
            def _register(handler):
                for fn in fns:
                    assert fn not in handlers, fn
                    handlers[fn] = handler
                return handler

            assert callable(fns[0])
            return _register

        from torch.backends.cuda import SDPAParams

        from . import (
            ConstantVariable,
            DeterministicAlgorithmsVariable,
            GradModeVariable,
            StreamContextVariable,
            SymNodeVariable,
            TensorVariable,
            UserDefinedObjectVariable,
        )
        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls

        @register(*tracing_state_functions)
        def handle_tracing_state_functions(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            assert not args and not kwargs
            # See: https://github.com/pytorch/pytorch/issues/110765
            if self.value in (
                torch._utils.is_compiling,
                torch._dynamo.external_utils.is_compiling,
                torch.compiler.is_compiling,
                torch.compiler.is_dynamo_compiling,
            ):
                tx.mark_inconsistent_side_effects()
            return ConstantVariable.create(tracing_state_functions[self.value])

        @register(torch.overrides.get_default_nowrap_functions.__wrapped__)
        def handle_get_default_nowrap_functions(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            # [Note: __torch_function__] we return empty here because we restrict
            # the set of functions that we trace __torch_function__ on to
            # functions outside of the actual set. Implementing this properly will require implementing
            # some variable types to track and compare tensor getset descriptors
            return VariableTracker.build(
                tx, torch.overrides.get_default_nowrap_functions()
            )

        @register(torch.ops.inductor.accumulate_grad_.default)
        def handle_accumulate_grad_(self, tx: "InstructionTranslator", *args, **kwargs):
            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.accumulate_grad), args, kwargs
            )

        @register(math.radians)
        def handle_radians(self, tx: "InstructionTranslator", *args, **kwargs):
            if not check_unspec_or_constant_args(args, kwargs):
                # Use polyfill to convert math.radians(x) into math.pi * x / 180.0
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.radians), args, kwargs
                )

        @register(torch.is_tensor, torch.overrides.is_tensor_like)
        def handle_is_tensor(self, tx: "InstructionTranslator", arg):
            if isinstance(arg, TensorVariable) or (
                self.value is torch.overrides.is_tensor_like
                and isinstance(arg, UserDefinedObjectVariable)
                and hasattr(arg.value, "__torch_function__")
            ):
                return ConstantVariable.create(True)
            else:
                return ConstantVariable.create(False)

        @register(
            torch.is_floating_point,
            torch.is_complex,
        )
        def handle_is_floating_point(self, tx: "InstructionTranslator", input):
            input_arg = input
            if isinstance(input_arg, TensorVariable) and input_arg.dtype is not None:
                if self.value is torch.is_floating_point:
                    return ConstantVariable.create(input_arg.dtype.is_floating_point)
                elif self.value is torch.is_complex:
                    return ConstantVariable.create(input_arg.dtype.is_complex)
                else:
                    raise AssertionError(f"calling {self.value}")

        @register(torch.numel)
        def handle_numel(self, tx: "InstructionTranslator", input):
            if isinstance(input, TensorVariable) and input.valid_size():
                return ConstantVariable.create(product(input.size))
            elif isinstance(input, TensorVariable):
                # Workaround dynamic shapes issue
                return input.call_method(tx, "numel", [], {})

        @register(torch.compile)
        def handle_torch_compile(self, tx: "InstructionTranslator", *args, **kwargs):
            if len(args) == 1:
                # torch.compile is a no-op in dynamo
                return args[0]

            unimplemented("torch.compile is used as a decorator in the compiled frame")

        @register(*REWRITE_OPS_TO_TENSOR_SIZE_METHOD)
        def handle_tensor_size_rewrites(self, tx: "InstructionTranslator", input):
            assert isinstance(input, TensorVariable)
            return input.call_method(tx, "size", [], {})

        @register(
            torch.nn.modules.utils._single,
            torch.nn.modules.utils._pair,
            torch.nn.modules.utils._triple,
            torch.nn.modules.utils._quadruple,
            torch.nn.modules.utils._ntuple,
        )
        def handle_ntuple(self, tx: "InstructionTranslator", *args, **kwargs):
            return self._call_ntuple(tx, args, kwargs)

        @register(torch.is_grad_enabled)
        def handle_is_grad_enabled(self, tx):
            install_guard(GradModeVariable._guards_singleton)
            return ConstantVariable.create(torch.is_grad_enabled())

        @register(torch.use_deterministic_algorithms)
        def handle_use_deterministic_algorithms(
            self, tx: "InstructionTranslator", mode, warn_only=False
        ):
            if warn_only and warn_only.as_python_constant():
                unimplemented("torch.use_deterministic_algorithms(warn_only=True)")
            return DeterministicAlgorithmsVariable.create(tx, mode.as_python_constant())

        @register(torch.are_deterministic_algorithms_enabled)
        def handle_are_deterministic_algorithms_enabled(self, tx):
            install_guard(DeterministicAlgorithmsVariable._guards_singleton)
            return ConstantVariable.create(torch.are_deterministic_algorithms_enabled())

        @register(torch._C._is_torch_function_enabled)
        def handle_is_torch_function_enabled(self, tx):
            install_guard(TorchFunctionDisableVariable._guards_singleton)
            return ConstantVariable.create(tx.output.torch_function_enabled)

        @register(
            torch.overrides.has_torch_function,
            torch.overrides.has_torch_function_variadic,
            torch.overrides.has_torch_function_unary,
        )
        def handle_has_torch_function(self, tx: "InstructionTranslator", *args):
            elems = (
                args[0].unpack_var_sequence(tx)
                if len(args) == 1 and isinstance(args[0], TupleVariable)
                else args
            )
            return ConstantVariable.create(
                any(has_torch_function(x) for x in elems),
            )

        @register(
            *dict.fromkeys(  # remove duplicates
                device_interface.stream
                for _, device_interface in get_registered_device_interfaces()
            )
        )
        def handle_device_interface_stream(self, tx: "InstructionTranslator", stream):
            return StreamContextVariable.create(tx, stream)

        @register(torch.from_numpy)
        def handle_from_numpy(self, tx: "InstructionTranslator", *args):
            if not config.trace_numpy:
                unimplemented("torch.from_numpy. config.trace_numpy is False")
            if not np:
                unimplemented("torch.from_numpy. NumPy is not available")
            return wrap_fx_proxy_cls(
                target_cls=TensorVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch.as_tensor,
                    *proxy_args_kwargs(args, {}),
                ),
                example_value=None,
            )

        @register(torch.jit.annotate)
        def handle_jit_annotate(self, tx: "InstructionTranslator", the_type, the_value):
            return the_value

        @register(torch.backends.cudnn.is_acceptable)
        def handle_cudnn_is_acceptable(
            self, tx: "InstructionTranslator", tensor, *extra
        ):
            # is_acceptable(tensor) returns true if
            #   (a) tensor dtype/device are supported by cudnn
            #   (b) cudnn is available
            #   (c) some initialization has completed
            # technically, it depends on some global state from (c) (torch.backends.cudnn.__cudnn_version)
            assert not extra, "Expect 1 input to cudnn.is_acceptable"
            assert isinstance(
                tensor, TensorVariable
            ), "Expect input to cudnn.is_acceptable to be a tensor"
            tensor_inp = torch.tensor(0, dtype=tensor.dtype, device=tensor.device)
            return ConstantVariable.create(
                torch.backends.cudnn.is_acceptable(tensor_inp)
            )

        @register(torch.utils.hooks.BackwardHook)
        def handle_backward_hook(self, tx: "InstructionTranslator", *args, **kwargs):
            return variables.BackwardHookVariable.create(tx, *args, **kwargs)

        @register(torch.nn.Parameter)
        def handle_parameter(self, tx: "InstructionTranslator", *args, **kwargs):
            return self.call_nn_parameter(tx, *args, **kwargs)

        @register(torch.ops.aten.sym_size, torch.ops.aten.sym_size.int)
        def handle_sym_size(self_, tx, self, dim=None):
            # we see this when retracing already traced code
            if dim is not None:
                return self.call_method(tx, "size", [dim], {})

        @register(torch.ops.aten.sym_stride, torch.ops.aten.sym_stride.int)
        def handle_sym_stride(self_, tx, self, dim=None):
            if dim is not None:
                return self.call_method(tx, "stride", [dim], {})

        @register(torch.addcdiv)
        def handle_addcdiv(self, tx: "InstructionTranslator", *args, **kwargs):
            if len(args) == 3 and "value" in kwargs and len(kwargs) == 1:
                # decompose addcdiv into constituent ops, prevents a graph break due to converting
                # value to a scalar
                result = TorchInGraphFunctionVariable(torch.div).call_function(
                    tx, [*args[1:]], {}
                )
                result = TorchInGraphFunctionVariable(torch.mul).call_function(
                    tx, [result, kwargs["value"]], {}
                )
                return TorchInGraphFunctionVariable(torch.add).call_function(
                    tx, [args[0], result], {}
                )

        @register(torch._foreach_lerp_)
        def handle_inplace_foreach_lerp_scalar(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            if len(args) == 3 and not isinstance(args[2], ListVariable) and not kwargs:
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.foreach_lerp_inplace),
                    args,
                    kwargs,
                )

        @register(torch._foreach_pow)
        def handle_foreach_pow_scalar(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            # In eager it's more performant to call item() from within the C op implementation
            # in compile, it's more performant to not graph break.
            if len(args) == 2 and isinstance(args[0], TensorVariable) and not kwargs:
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.foreach_pow_scalar),
                    args,
                    kwargs,
                )

        @register(torch._assert)
        def handle_assert(self, tx: "InstructionTranslator", condition, message):
            if (condition.is_python_constant() and condition.as_python_constant()) or (
                isinstance(condition, variables.SymNodeVariable)
                and condition.evaluate_expr()
            ):
                return ConstantVariable(None)

        @register(SDPAParams)
        def handle_sdpa_params(self, tx: "InstructionTranslator", *args, **kwargs):
            return wrap_fx_proxy(
                tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch._C._SDPAParams,
                    *proxy_args_kwargs(args, kwargs),
                ),
                param_vars=args,
            )

        if DistributedVariable.is_available():
            from torch.distributed.distributed_c10d import (
                _get_group_size_by_name,
                _get_group_tag,
                _rank_not_in_group,
                _resolve_group_name_by_ranks_and_tag,
                get_process_group_ranks,
            )
            from torch.distributed.tensor import DTensor

            @register(
                _get_group_size_by_name,
                _get_group_tag,
                _rank_not_in_group,
                get_process_group_ranks,
                _resolve_group_name_by_ranks_and_tag,
            )
            def handle_constant_processgroup_functions(
                self, tx: "InstructionTranslator", *args
            ):
                # because the input is a "ProcessGroupVariable", we'll be guarding on its
                # ID_MATCH based on how it was constructed.

                # We desugar it at trace-time into ranks by directly calling util
                # bake the result into the trace
                if len(args) == 1:
                    # group or group name
                    assert isinstance(args[0], (ProcessGroupVariable, ConstantVariable))
                elif len(args) == 2:
                    # ranks + tag
                    assert isinstance(args[0], ListVariable) and isinstance(
                        args[1], ConstantVariable
                    )
                else:
                    raise AssertionError(
                        f"Invalid group value ({args}) for constant pg "
                        f"function {self.value}"
                    )
                args_as_value = [arg.as_python_constant() for arg in args]
                invocation_result = self.value(*args_as_value)

                # Note - while we *could* cook up sources around invocations, like a FunctionSource
                # the space of invoking functions in the middle of the guard chain is very iffy. As such,
                # guard propagation via options is the best we can do.
                return VariableTracker.build(tx, invocation_result)

            @register(DTensor.from_local)
            def handle_from_local(self, tx: "InstructionTranslator", *args, **kwargs):
                # rewrite non-primitive args/kwargs to be included in the on-the-fly prim function
                # and rewrite args to have only proxyable args, then insert call_function
                args_as_value = [x.as_python_constant() for x in args[1:]]
                kwargs_as_value = {
                    k: v.as_python_constant()
                    for k, v in kwargs.items()
                    if k not in ["shape", "stride"]
                }
                kwargs_to_be_proxied = {
                    k: kwargs[k] for k in ["shape", "stride"] if k in kwargs
                }

                def fn_with_prim_types(x, shape=None, stride=None):
                    return self.value(
                        x, *args_as_value, **kwargs_as_value, shape=shape, stride=stride
                    )

                # attach the same function name for better debugging
                fn_with_prim_types.__name__ = "prim " + self.value.__name__

                return wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        fn_with_prim_types,
                        *proxy_args_kwargs(
                            [args[0]],
                            kwargs_to_be_proxied,
                        ),
                    ),
                )

        @register(torch.nested.nested_tensor)
        def handle_nested_tensor(
            self,
            tx: "InstructionTranslator",
            tensor_list=None,
            *args,
            layout=None,
            **kwargs,
        ):
            from .lists import BaseListVariable

            if layout and layout.as_python_constant() == torch.strided:
                unimplemented("torch.compile does not support strided NestedTensor")
            if not isinstance(tensor_list, BaseListVariable):
                unimplemented("nested_tensor with non-list input")

        @register(torch.nn.functional.one_hot)
        def handle_one_hot(self, tx: "InstructionTranslator", *args, **kwargs):
            if len(args) + len(kwargs) == 1 or (
                len(args) == 2
                and args[1].is_python_constant()
                and args[1].as_python_constant() == -1
            ):
                unimplemented(
                    "torch.nn.functional.one_hot with data-dependent output shape"
                )

        @register(torch.fx.experimental.symbolic_shapes.guard_size_oblivious)
        def handle_guard_size_oblivious(self, tx: "InstructionTranslator", expr):
            if isinstance(expr, SymNodeVariable):
                # TODO: this probably should be folded somewhere else but I'm not sure where
                # TODO: some of the other symbolic_shapes special tools can also get this treatment too
                return variables.ConstantVariable.create(
                    torch.fx.experimental.symbolic_shapes.guard_size_oblivious(
                        expr.sym_num
                    )
                )
            elif isinstance(expr, ConstantVariable):
                return expr

        @register(torch._C._autograd._unsafe_set_version_counter)
        def handle_unsafe_set_version_counter(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            from ..tensor_version_op import _unsafe_set_version_counter

            return TorchInGraphFunctionVariable(
                _unsafe_set_version_counter
            ).call_function(tx, [*args], kwargs)

        @register(torch.tensor)
        def handle_torch_tensor(self, tx: "InstructionTranslator", *args, **kwargs):
            def check_any_unspec(x):
                # NB: This includes UnspecializedPythonVariable
                if isinstance(x, (TensorVariable, SymNodeVariable)):
                    return True
                elif isinstance(x, (ListVariable, TupleVariable)):
                    return any(check_any_unspec(y) for y in x.items)
                # TODO: there maybe other recursive structures you need to
                # check
                else:
                    return False

            data_arg = None
            if args:
                data_arg = args[0]
            elif "data" in kwargs:
                data_arg = kwargs["data"]

            # NB: OK to pass torch.tensor(tensor), this will trace fine
            if not isinstance(data_arg, TensorVariable) and check_any_unspec(data_arg):
                # This is slower and less canonical, so only use it if we
                # have to
                return TorchInGraphFunctionVariable(torch._refs.tensor).call_function(
                    tx, [*args], kwargs
                )

        @register(torch._C._pop_torch_function_stack)
        def handle_pop_torch_function(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            assert not args and not kwargs
            if not tx.symbolic_torch_function_state.mode_stack:
                raise unimplemented("Popping from an empty torch function mode stack")
            TorchFunctionModeStackVariable.register_mutation(tx)
            return tx.symbolic_torch_function_state.pop_torch_function_mode()

        @register(torch._C._push_on_torch_function_stack)
        def handle_push_torch_function(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            assert len(args) == 1 and not kwargs
            TorchFunctionModeStackVariable.register_mutation(tx)
            tx.symbolic_torch_function_state.push_torch_function_mode(args[0])
            return ConstantVariable.create(None)

        @register(torch._C._len_torch_function_stack)
        def handle_len_torch_function(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            assert not args and not kwargs
            return ConstantVariable.create(
                len(tx.symbolic_torch_function_state.mode_stack)
            )

        @register(torch._C._get_function_stack_at)
        def handle_get_stack_at(self, tx: "InstructionTranslator", *args, **kwargs):
            assert len(args) == 1 and not kwargs
            ind = args[0].as_python_constant()
            assert ind >= 0 and ind < len(tx.symbolic_torch_function_state.mode_stack)
            return tx.symbolic_torch_function_state.mode_stack[ind]

        @register(torch.set_default_device)
        def handle_set_default_device(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            # Today this is inserted in the graph, once TF mode
            # handling is complete, we can trace the device context
            # like any other TF mode and remove this special handling
            # Insert the TF mode representing the device context at
            # the bottom of the stack to match the eager semantics
            # Running the graph will ensure that the DeviceContext mode is
            # at the correct position in the stack
            TorchFunctionModeStackVariable.register_mutation(tx)
            if args[0].is_python_constant() and args[0].as_python_constant() is None:
                TorchFunctionModeStackVariable.clear_default_device(tx)
            else:
                TorchFunctionModeStackVariable.register_device_context_insertion(tx)

            return ConstantVariable.create(None)

        return handlers

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ConstantVariable, SymNodeVariable, TensorVariable
        from .builder import wrap_fx_proxy

        if self.torch_function_override_enabled(tx, args, kwargs):
            return dispatch_torch_function(tx, self, args, kwargs)

        if self.can_constant_fold_through() and check_unspec_or_constant_args(
            args, kwargs
        ):
            # constant fold functions need to be guarded.
            if self.value in constant_fold_functions_need_guards:
                source = CallFunctionNoArgsSource(self.source)
                install_guard(source.make_guard(GuardBuilder.EQUALS_MATCH))
            # constant fold
            return ConstantVariable.create(
                self.as_python_constant()(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
            )

        if self.is_tensor_method():
            return self.call_tensor_method(tx, args, kwargs)

        special_handler = self._get_handlers().get(self.value)
        if special_handler:
            result = special_handler(self, tx, *args, **kwargs)
            if result:
                return result

        any_symints_or_symfloats = any(isinstance(x, SymNodeVariable) for x in args)

        all_ints_or_floats = all(
            isinstance(x, (variables.ConstantVariable, variables.SymNodeVariable))
            for x in args
        )
        if (
            getattr(self.value, "__module__", "") == "torch"
            and self.value.__name__ in bin_ops
            and any_symints_or_symfloats
            and all_ints_or_floats
        ):
            msg = f"""\
Calling {str(self.value)} on only torch.SymInt arguments is not yet supported.
To support this behavior, we need to allow const-propping tensors that store symint data.
For now, dynamo will explicitly graph break when it encounters user code with this behavior.
"""
            log.warning(msg)
            unimplemented(msg)

        # TODO(voz): Replace w/ dynamic shape rewrite table.
        # Ideally, we would be able to do this at ctor time, but alas we need a combination
        # of value + args to determine this.
        fn_ = self.value
        if any_symints_or_symfloats:
            torch_sym_op = f"_sym_{self.value.__name__}"
            if getattr(self.value, "__module__", None) == "math" and hasattr(
                torch, torch_sym_op
            ):
                fn_ = getattr(torch, torch_sym_op)

        fake_out_shape = None
        if "out" in kwargs and isinstance(kwargs["out"], variables.TensorVariable):
            # Calling fake tensor propagation can mutate the out= tensor in
            # tx.output.tracked_fakes. tracked_fakes are used to apply
            # symbolic_shape guards. Mutating them destroys the information
            # prior to tracing, which is essential for creating right
            # guards. So save the shape now, and check later if it has
            # changed. If it has, graph break.
            fake_out_shape = kwargs["out"].proxy.node.meta["example_value"].shape

        tensor_variable = wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                fn_,
                *proxy_args_kwargs(args, kwargs),
            ),
        )

        if (
            isinstance(tensor_variable, TensorVariable)
            and "requires_grad" in kwargs
            and kwargs["requires_grad"].as_python_constant()
        ):
            unimplemented(
                """factory functions that return tensors that require grad are not supported.
Either create the tensor outside the compiled region, or do not set the tensor to require_grad"""
            )

        if "out" in kwargs and not (
            isinstance(kwargs["out"], variables.ConstantVariable)
            and kwargs["out"].as_python_constant() is None
        ):
            # out variants of torch operators like torch.sort and torch.sigmoid
            # mutate the tensors in the out field.
            #
            # However, it's non-trivial to update all references of the old
            # `TensorVariable` to the new one returned (`result_var`), so we
            # take the conservative approach to graph break on size changes, and
            # assume other cases can fall through soundly.
            #
            # Note that although these tensor variablels would hold different
            # proxies, the in-place mutation semantics is preserved in the FX
            # graph, so we won't have correctness issues.
            if isinstance(tensor_variable, TupleVariable):
                assert isinstance(kwargs["out"], (TupleVariable, ListVariable))
                for out_tensor, result_tensor in zip(
                    kwargs["out"].items, tensor_variable.items
                ):
                    if (
                        isinstance(out_tensor, variables.TensorVariable)
                        and isinstance(result_tensor, variables.TensorVariable)
                        and out_tensor._size
                        != result_tensor._size  # we actually want to compare None values here
                    ):
                        # It's hard to get out variants with resizing on graph inputs work
                        # properly across dynamo/aot/inductor, just fall back.
                        unimplemented("out variants with resizing on graph inputs")
            elif isinstance(tensor_variable, TensorVariable):
                assert isinstance(kwargs["out"], TensorVariable)
                assert "example_value" in kwargs["out"].proxy.node.meta
                fake_tensor = tensor_variable.proxy.node.meta["example_value"]
                fake_out = kwargs["out"].proxy.node.meta["example_value"]
                if fake_out_shape != fake_tensor.shape:
                    # It's hard to get out variants with resizing on graph inputs work
                    # properly across dynamo/aot/inductor, just fall back.
                    unimplemented("out variants with resizing on graph inputs")
                if not torch._prims_common.is_contiguous(fake_out):
                    # It's difficult to handle strides correctly in functionalization
                    # when calling an out= op with a non-contiguous out argument
                    unimplemented(
                        "out= op was called where output tensor was non-contiguous"
                    )
            elif (
                isinstance(tensor_variable, ConstantVariable)
                and tensor_variable.value is None
            ):
                # Handle out-variant custom ops that return None.
                if isinstance(kwargs["out"], TensorVariable):
                    assert "example_value" in kwargs["out"].proxy.node.meta
                    fake_out = kwargs["out"].proxy.node.meta["example_value"]
                    if not torch._prims_common.is_contiguous(fake_out):
                        # It's difficult to handle strides correctly in functionalization
                        # when calling an out= op with a non-contiguous out argument
                        unimplemented(
                            "out= op was called where output tensor was non-contiguous"
                        )
                elif isinstance(kwargs["out"], ListVariable):
                    for idx, x in enumerate(kwargs["out"].items):
                        assert "example_value" in x.proxy.node.meta  # type: ignore[attr-defined]
                        fake_out = x.proxy.node.meta["example_value"]  # type: ignore[attr-defined]
                        if not torch._prims_common.is_contiguous(fake_out):
                            # It's difficult to handle strides correctly in functionalization
                            # when calling an out= op with a non-contiguous out argument
                            unimplemented(
                                "out= op was called where some of the output tensors were non-contiguous"
                            )
            else:
                unimplemented(f"out variant of {type(kwargs['out'])}")

        return tensor_variable

    def _call_ntuple(self, tx: "InstructionTranslator", args, kwargs):
        """inline behavior of torch.nn.modules.utils._ntuple"""
        if self.value is torch.nn.modules.utils._ntuple:
            count = args[0].as_python_constant()
        else:
            count = self.value.__closure__[0].cell_contents
        assert isinstance(count, int)
        assert not kwargs

        def handle_ntuple(value):
            if value.has_unpack_var_sequence(tx):
                return variables.TupleVariable(
                    list(value.unpack_var_sequence(tx)),
                )
            elif value.is_python_constant():
                # constant prop through it
                return variables.ConstantVariable.create(
                    torch.nn.modules.utils._ntuple(count)(value.as_python_constant()),
                )
            else:
                unimplemented(f"torch.nn.modules.utils._ntuple({value})")

        if self.value is torch.nn.modules.utils._ntuple:
            return variables.LambdaVariable(handle_ntuple)
        else:
            return handle_ntuple(args[0])

    @classmethod
    def call_nn_parameter(cls, tx, data=None, requires_grad=True):
        """A call to torch.nn.Parameter() gets lifted to before the graph"""
        if tx.export:
            unimplemented("nn parameter construction not supported with export")

        if isinstance(requires_grad, variables.VariableTracker):
            try:
                requires_grad = requires_grad.as_python_constant()
            except NotImplementedError:
                unimplemented("Parameter(requires_grad=...) not constant")

        if not isinstance(data, variables.TensorVariable):
            unimplemented(f"Parameter(data={data}) not implemented")

        # this results in cleaner graphs, but only works for inputs
        if data.source:
            return cls._nn_param_via_prefix_insert(tx, data, requires_grad)

        if is_traceable_wrapper_subclass_type(data.class_type):
            unimplemented("Parameter constructor with tensor subclass NYI")

        if not can_convert_to_tracable_parameter():
            unimplemented("Workaround for issues with nn_parameter construction")

        try:
            shape = tuple(data.var_getattr(tx, "shape").as_python_constant())
            dtype = data.var_getattr(tx, "dtype").as_python_constant()
            device = data.var_getattr(tx, "device").as_python_constant()
        except NotImplementedError as e:
            unimplemented(f"Parameter not python_constant: {e}")

        placeholder = tx.output.synthetic_graph_input(
            new_parameter_placeholder, [shape, dtype, device, requires_grad]
        )
        if data.requires_grad:
            data = data.call_method(tx, "detach", [], {})

        from .builder import wrap_fx_proxy

        result = wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_function",
                tracable_create_parameter,
                (data.as_proxy(), placeholder.as_proxy()),
                {},
            ),
            # In reconstruct() we should use the original parameter. The one
            # returned by the graph will be an alias.
            source=placeholder.source,
        )
        assert isinstance(result, variables.TensorVariable)
        result.class_type = torch.nn.Parameter

        # TODO(jansel/bdhirsh) - There is some issue with
        # tracable_create_paramter. It does not seem to use the right
        # grad_enabled. Since this is parameter, we can just override the
        # has_grad_fn field to False to workaround the issue.
        result.has_grad_fn = False

        # TODO(jansel): if the new param falls out of scope, currently it won't get freed until
        # the end of the graph.  We should fix this.
        return result

    @staticmethod
    def _nn_param_via_prefix_insert(tx: "InstructionTranslator", data, requires_grad):
        # Alternate version if we have a .source
        varname = tx.output.new_var()

        # construct the nn.Parmeter before the graph save it to varname
        cg = PyCodegen(tx)
        cg.add_push_null(lambda: cg.load_import_from("torch.nn", "Parameter"))
        cg(data.source)
        cg(variables.ConstantVariable(requires_grad))
        cg.call_function(2, False)
        cg.store(varname)
        tx.output.pregraph_bytecode.extend(cg.get_instructions())

        data_node = data.as_proxy().node
        if data_node.op not in ("placeholder", "get_attr"):
            unimplemented(
                "Unexpected type of data placeholder op for parameter construction"
            )

        # add the newly constructed nn.Parameter as a graph input
        source = SyntheticLocalSource(varname)
        example_value = torch.nn.Parameter(
            tx.output.example_value_from_input_node(data.as_proxy().node)
        )
        result = VariableTracker.build(tx, example_value, source)
        # No need to guard on this since we already guarded on `data`.
        # These guards would fail since varname doesn't exist until after the function starts
        TracingContext.get().guards_context.dynamo_guards.remove_guards_with_source(
            source
        )
        return result

    def call_tensor_method(self, tx, args, kwargs):
        return args[0].call_method(tx, self.get_function().__name__, args[1:], kwargs)

    def is_tensor_method(self):
        return (
            inspect.ismethoddescriptor(self.get_function())
            and hasattr(self.get_function(), "__objclass__")
            and self.get_function().__objclass__ == torch._C.TensorBase
        ) or self.get_function() is torch.Tensor.__contains__

    def torch_function_override_enabled(self, tx, args, kwargs):
        return (
            self.get_function() in get_overridable_functions()
            or isinstance(
                self.get_function(),
                (torch._ops.OpOverload, torch._ops.OpOverloadPacket),
            )
        ) and can_dispatch_torch_function(tx, args, kwargs)
