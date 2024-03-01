# mypy: ignore-errors

import inspect
import logging

import math
import re
from typing import Dict, List

import torch._C
import torch._refs
import torch.fx
import torch.nn
import torch.onnx.operators
from torch._logging import warning_once

from torch._streambase import _StreamBase
from ..._guards import TracingContext
from .. import config, polyfill, variables
from ..codegen import PyCodegen
from ..device_interface import get_registered_device_interfaces
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import LocalSource
from ..utils import (
    check_constant_args,
    check_unspec_python_args,
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
from .distributed import is_constant_pg_functions, is_from_local, ProcessGroupVariable
from .lists import ListVariable, TupleVariable
from .torch_function import can_dispatch_torch_function, dispatch_torch_function

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

log = logging.getLogger(__name__)

supported_ctx_manager_classes = {
    torch.profiler.profiler.profile,
    torch.autograd.profiler.profile,
    torch.autograd.profiler.record_function,
    torch._C.DisableTorchFunctionSubclass,
    torch._functorch.vmap.vmap_increment_nesting,
    torch._functorch.eager_transforms.grad_increment_nesting,
    torch._functorch.eager_transforms.enable_inplace_requires_grad,
    torch.amp.autocast_mode.autocast,
    torch.autograd.grad_mode.enable_grad,
    torch.autograd.grad_mode.inference_mode,
    torch.autograd.grad_mode.no_grad,
    torch.autograd.grad_mode.set_grad_enabled,
    torch.autograd.graph.disable_saved_tensors_hooks,
    torch.cpu.amp.autocast_mode.autocast,
    torch.cuda.amp.autocast_mode.autocast,
}


REWRITE_OPS_TO_TENSOR_SIZE_METHOD = [
    torch.onnx.operators.shape_as_tensor,
    torch._shape_as_tensor,
]

constant_fold_functions = [
    torch._assert,
    torch._utils._get_device_index,
    torch._C._get_cublas_allow_tf32,
    torch.cuda.get_device_properties,
    torch.cuda.is_available,
    torch.distributed.is_available,
    torch.get_autocast_gpu_dtype,
    torch.get_default_dtype,
    torch.is_autocast_cache_enabled,
    torch.is_autocast_cpu_enabled,
    torch.is_autocast_enabled,
    torch.is_complex,
    torch.is_floating_point,
    torch.nn.functional._Reduction.get_enum,
    torch.promote_types,
    torch._C._get_privateuse1_backend_name,
]


if torch.distributed.is_available():
    constant_fold_functions.extend(
        [
            torch.distributed.is_initialized,
            torch.distributed.get_rank,
            torch.distributed.get_world_size,
        ]
    )


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
}


class BaseTorchVariable(VariableTracker):
    """common base for all torch.* functions, classes, modules and other things"""

    @classmethod
    def create_with_source(cls, value, source):
        install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        return cls(
            value,
            source=source,
        )

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def reconstruct(self, codegen):
        try:
            name = f"{self.value.__module__}.{self.value.__name__}"
        except Exception:
            name = f"torch_obj_{id(self.value)}"
        unique_var_name = "__" + re.sub(r"[^a-zA-Z0-9_]+", "_", name)
        codegen.extend_output(
            codegen.setup_globally_cached(unique_var_name, self.value, False)
        )

    def as_proxy(self):
        return self.value

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    def call_hasattr(self, tx, name):
        result = hasattr(self.value, name)
        return variables.ConstantVariable.create(result)

    def can_constant_fold_through(self):
        if self.value in constant_fold_functions:
            return True
        return getattr(self.value, "__module__", None) == "math"


class TorchCtxManagerClassVariable(BaseTorchVariable):
    """Points to a context manager class in torch.* that dynamo has implementations"""

    def __repr__(self):
        return f"TorchCtxManagerClassVariable({self.value})"

    @staticmethod
    def is_matching_cls(value):
        # Unwrap if it's a functools.lru_cache wrapper
        value = unwrap_if_wrapper(value)
        # We can't do isinstance(value, type) check because some ctx managers
        # are implemented as a function decorated by contextlib.contextmanager,
        # E.g., torch._functorch.vmap.vmap_increment_nesting.
        return hashable(value) and value in supported_ctx_manager_classes

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import (
            DisabledSavedTensorsHooksVariable,
            GradIncrementNestingCtxManagerVariable,
            GradInplaceRequiresGradCtxManagerVariable,
            GradModeVariable,
            InferenceModeVariable,
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
        elif inspect.isclass(self.value) and issubclass(self.value, _StreamBase):
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
                [guard_if_dyn(x) for x in args],
            )
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


class TorchInGraphFunctionVariable(BaseTorchVariable):
    """Points to a torch function/method that should be put in FX graph"""

    def __repr__(self):
        return f"TorchInGraphFunctionVariable({self.value})"

    def get_function(self):
        return self.value

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import (
            ConstantVariable,
            DeterministicAlgorithmsVariable,
            GradModeVariable,
            SDPAParamsVariable,
            StreamContextVariable,
            SymNodeVariable,
            TensorVariable,
            UserDefinedObjectVariable,
        )

        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls

        constant_args = check_constant_args(args, kwargs)
        unspec_python_args = check_unspec_python_args(args, kwargs)

        if self.can_constant_fold_through() and (constant_args or unspec_python_args):
            # constant fold
            return ConstantVariable.create(
                self.as_python_constant()(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
            )
        elif self.value in tracing_state_functions:
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
        elif self.value is torch.overrides.get_default_nowrap_functions.__wrapped__:
            # [Note: __torch_function__] we return empty here because we restrict
            # the set of functions that we trace __torch_function__ on to
            # functions outside of the actual set. Implementing this properly will require implementing
            # some variable types to track and compare tensor getset descriptors
            from .builder import SourcelessBuilder

            return SourcelessBuilder()(
                tx, torch.overrides.get_default_nowrap_functions()
            )
        elif self.value == torch.ops.inductor.accumulate_grad_.default:
            from .builder import SourcelessBuilder

            return tx.inline_user_function_return(
                SourcelessBuilder()(tx, polyfill.accumulate_grad), args, kwargs
            )
        elif self.value == math.radians and not (constant_args or unspec_python_args):
            # Use polyfill to convert math.radians(x) into math.pi * x / 180.0
            from .builder import SourcelessBuilder

            return tx.inline_user_function_return(
                SourcelessBuilder()(tx, polyfill.radians), args, kwargs
            )
        elif self.value in (torch.is_tensor, torch.overrides.is_tensor_like):
            assert len(args) == 1
            if isinstance(args[0], TensorVariable) or (
                self.value is torch.overrides.is_tensor_like
                and isinstance(args[0], UserDefinedObjectVariable)
                and hasattr(args[0].value, "__torch_function__")
            ):
                return ConstantVariable.create(True)
            else:
                return ConstantVariable.create(False)
        elif self.value in (
            torch.is_floating_point,
            torch.is_complex,
        ):
            input_arg = None
            if args:
                input_arg = args[0]
            else:
                assert "input" in kwargs
                input_arg = kwargs["input"]
            if isinstance(input_arg, TensorVariable) and input_arg.dtype is not None:
                if self.value is torch.is_floating_point:
                    return ConstantVariable.create(input_arg.dtype.is_floating_point)
                elif self.value is torch.is_complex:
                    return ConstantVariable.create(input_arg.dtype.is_complex)
                else:
                    raise AssertionError(f"calling {self.value}")
        elif (
            self.value is torch.numel
            and isinstance(args[0], TensorVariable)
            and args[0].size is not None
        ):
            return ConstantVariable.create(product(args[0].size))
        elif self.value in REWRITE_OPS_TO_TENSOR_SIZE_METHOD:
            assert len(args) == 1
            assert isinstance(args[0], TensorVariable)
            return args[0].call_method(tx, "size", [], {})
        elif self.value in (
            torch.nn.modules.utils._single,
            torch.nn.modules.utils._pair,
            torch.nn.modules.utils._triple,
            torch.nn.modules.utils._quadruple,
            torch.nn.modules.utils._ntuple,
        ):
            return self._call_ntuple(tx, args, kwargs)
        elif self.value is torch.is_grad_enabled:
            assert not (args or kwargs)
            install_guard(GradModeVariable._guards_singleton)
            return ConstantVariable.create(torch.is_grad_enabled())
        elif self.value is torch.use_deterministic_algorithms and len(args) == 1:
            return DeterministicAlgorithmsVariable.create(
                tx, args[0].as_python_constant()
            )
        elif self.value is torch.are_deterministic_algorithms_enabled:
            assert not (args or kwargs)
            install_guard(DeterministicAlgorithmsVariable._guards_singleton)
            return ConstantVariable.create(torch.are_deterministic_algorithms_enabled())
        elif self.value is torch._C._is_torch_function_enabled:
            assert not (args or kwargs)
            install_guard(TorchFunctionDisableVariable._guards_singleton)
            return ConstantVariable.create(tx.output.torch_function_enabled)
        elif self.value in (
            torch.overrides.has_torch_function,
            torch.overrides.has_torch_function_variadic,
            torch.overrides.has_torch_function_unary,
        ):
            assert not kwargs
            return ConstantVariable.create(
                any(has_torch_function(a) for a in args),
            )
        elif any(
            self.value is method
            for method in [
                device_interface.stream
                for _, device_interface in get_registered_device_interfaces()
            ]
        ):
            assert len(args) == 1
            return StreamContextVariable.create(tx, args[0])
        elif self.value is torch.from_numpy:
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
        elif can_dispatch_torch_function(tx, args, kwargs):
            return dispatch_torch_function(tx, self, args, kwargs)
        elif self.value is torch.jit.annotate:
            assert len(args) == 2
            return args[1]
        elif self.value is torch.backends.cudnn.is_acceptable:
            # is_acceptable(tensor) returns true if
            #   (a) tensor dtype/device are supported by cudnn
            #   (b) cudnn is available
            #   (c) some initialization has completed
            # technically, it depends on some global state from (c) (torch.backends.cudnn.__cudnn_version)
            assert (
                len(args) == 1 or "tensor" in kwargs
            ), "Expect 1 input to cudnn.is_acceptable"
            tensor_variable = args[0] if len(args) > 0 else kwargs["tensor"]
            assert isinstance(
                tensor_variable, TensorVariable
            ), "Expect input to cudnn.is_acceptable to be a tensor"
            tensor_inp = torch.tensor(
                0, dtype=tensor_variable.dtype, device=tensor_variable.device
            )
            return ConstantVariable.create(
                torch.backends.cudnn.is_acceptable(tensor_inp)
            )
        elif self.value is torch.utils.hooks.BackwardHook:
            return variables.BackwardHookVariable.create(tx, *args, **kwargs)
        elif self.value is torch.nn.Parameter:
            return self.call_nn_parameter(tx, *args, **kwargs)
        elif (
            self.value == torch.numel
            and len(args) == 1
            and isinstance(args[0], TensorVariable)
            and len(kwargs) == 0
        ):
            # TODO(voz): This is rewritten as a call_method because
            # torch.numel(x) w/ sym shapes raises a RuntimeError and x.numel() does not
            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method",
                    "numel",
                    *proxy_args_kwargs(args, kwargs),
                ),
            )
        # TODO: These special cases shouldn't be necessary; we should
        # generically support torch.ops that return int
        elif (
            self.value in (torch.ops.aten.sym_size, torch.ops.aten.sym_size.int)
            and len(args) == 2
            and len(kwargs) == 0
            and isinstance(args[0], TensorVariable)
        ):
            # we see this when retracing already traced code
            return args[0].call_method(tx, "size", [args[1]], {})
        elif (
            self.value in (torch.ops.aten.sym_stride, torch.ops.aten.sym_stride.int)
            and len(args) == 2
            and len(kwargs) == 0
            and isinstance(args[0], TensorVariable)
        ):
            return args[0].call_method(tx, "stride", [args[1]], {})
        elif (
            self.value == torch.addcdiv
            and len(args) == 3
            and "value" in kwargs
            and len(kwargs) == 1
        ):
            # decompose addcdiv into constituent ops, prevents a graph break due to converting
            # value to a scalar
            result = TorchInGraphFunctionVariable(torch.div).call_function(
                tx, args[1:], {}
            )
            result = TorchInGraphFunctionVariable(torch.mul).call_function(
                tx, [result, kwargs["value"]], {}
            )
            return TorchInGraphFunctionVariable(torch.add).call_function(
                tx, [args[0], result], {}
            )
        elif (
            self.value is torch._assert
            and len(args) >= 1
            and (
                (args[0].is_python_constant() and args[0].as_python_constant())
                or (
                    isinstance(args[0], variables.SymNodeVariable)
                    and args[0].evaluate_expr()
                )
            )
        ):
            return ConstantVariable(None)
        elif SDPAParamsVariable.is_sdpa_params(self.value):
            return wrap_fx_proxy(
                tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch._C._SDPAParams,
                    *proxy_args_kwargs(args, kwargs),
                ),
                param_vars=args,
            )
        elif is_constant_pg_functions(self.value):
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
            from .builder import SourcelessBuilder

            return SourcelessBuilder()(tx, invocation_result)
        elif is_from_local(self.value):
            # rewrite non-primitive args/kwargs to be included in the on-the-fly prim function
            # and rewrite args to have only proxyable args, then insert call_function
            args_as_value = [x.as_python_constant() for x in args[1:]]
            kwargs_as_value = {k: v.as_python_constant() for k, v in kwargs.items()}

            def fn_with_prim_types(x):
                return self.value(x, *args_as_value, **kwargs_as_value)

            # attach the same function name for better debugging
            fn_with_prim_types.__name__ = "prim " + self.value.__name__

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    fn_with_prim_types,
                    *proxy_args_kwargs([args[0]], {}),
                ),
            )
        elif (
            self.value is torch.nested.nested_tensor
            and kwargs.get("layout", torch.strided) == torch.strided
        ):
            raise unimplemented("torch.compile does not support strided NestedTensor")
        elif self.value is torch.nn.functional.one_hot and (
            len(args) + len(kwargs) == 1
            or (
                len(args) == 2
                and args[1].is_python_constant()
                and args[1].as_python_constant() == -1
            )
        ):
            raise unimplemented(
                "torch.nn.functional.one_hot with data-dependent output shape"
            )
        elif (
            self.value is torch.fx.experimental.symbolic_shapes.guard_size_oblivious
            and len(args) == 1
            and isinstance(args[0], SymNodeVariable)
        ):
            # TODO: this probably should be folded somewhere else but I'm not
            # sure where
            # TODO: some of the other symbolic_shapes special tools can also
            # get this treatment too
            (cond,) = args
            return variables.ConstantVariable.create(
                torch.fx.experimental.symbolic_shapes.guard_size_oblivious(cond.sym_num)
            )

        else:
            any_symints_or_symfloats = any(isinstance(x, SymNodeVariable) for x in args)
            all_ints_or_floats = all(
                isinstance(x, (variables.ConstantVariable, variables.SymNodeVariable))
                for x in args
            )
            bin_ops = {"add", "sub", "mul", "div", "sqrt"}
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
                raise unimplemented(msg)

            # TODO(voz): Replace w/ dynamic shape rewrite table.
            # Ideally, we would be able to do this at ctor time, but alas we need a combination
            # of value + args to determine this.
            fn_ = self.value
            if any(isinstance(x, SymNodeVariable) for x in args):
                torch_sym_op = f"_sym_{self.value.__name__}"
                if getattr(self.value, "__module__", None) == "math" and hasattr(
                    torch, torch_sym_op
                ):
                    fn_ = getattr(torch, torch_sym_op)

            if fn_ is torch.tensor:

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
                if not isinstance(data_arg, TensorVariable) and check_any_unspec(
                    data_arg
                ):
                    # This is slower and less canonical, so only use it if we
                    # have to
                    fn_ = torch._refs.tensor

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
                # out variants of torch operators like torch.sort and
                # torch.sigmoid mutate the tensors in the out field. Track such
                # tensors and rewrite the symbolic locals.
                if isinstance(tensor_variable, TupleVariable):
                    assert isinstance(kwargs["out"], (TupleVariable, ListVariable))
                    output_tensor_names = [
                        tx.find_symbolic_locals_name(x) for x in kwargs["out"].items
                    ]
                    for idx, name in enumerate(output_tensor_names):
                        if name in tx.symbolic_locals:
                            tx.symbolic_locals[name] = tensor_variable.items[idx]
                    for out_tensor, result_tensor in zip(
                        kwargs["out"].items, tensor_variable.items
                    ):
                        if (
                            out_tensor.source
                            and out_tensor in tx.output.graphargs
                            and out_tensor.size != result_tensor.size
                        ):
                            # It's hard to get out variants with resizing on graph inputs work
                            # properly across dynamo/aot/inductor, just fall back.
                            unimplemented("out variants with resizing on graph inputs")
                elif isinstance(tensor_variable, TensorVariable):
                    assert isinstance(kwargs["out"], TensorVariable)
                    assert "example_value" in kwargs["out"].proxy.node.meta
                    fake_tensor = tensor_variable.proxy.node.meta["example_value"]
                    fake_out = kwargs["out"].proxy.node.meta["example_value"]
                    if (
                        kwargs["out"].source
                        and kwargs["out"] in tx.output.graphargs
                        and fake_out.shape != fake_tensor.shape
                    ):
                        # It's hard to get out variants with resizing on graph inputs work
                        # properly across dynamo/aot/inductor, just fall back.
                        unimplemented("out variants with resizing on graph inputs")
                    if not torch._prims_common.is_contiguous(fake_out):
                        # It's difficult to handle strides correctly in functionalization
                        # when calling an out= op with a non-contiguous out argument
                        unimplemented(
                            "out= op was called where output tensor was non-contiguous"
                        )
                    name = tx.find_symbolic_locals_name(kwargs["out"])
                    if name in tx.symbolic_locals:
                        tx.symbolic_locals[name] = tensor_variable
                else:
                    unimplemented(f"out variant of {type(kwargs['out'])}")

            return tensor_variable

    def _call_ntuple(self, tx, args, kwargs):
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

        unimplemented("Parameter() on non-input")

    @staticmethod
    def _nn_param_via_prefix_insert(tx, data, requires_grad):
        # Alternate version if we have a .source
        from .builder import VariableBuilder

        varname = tx.output.new_var()

        # construct the nn.Parmeter before the graph save it to varname
        cg = PyCodegen(tx)
        cg.load_import_from("torch.nn", "Parameter")
        cg(data.source)
        cg(variables.ConstantVariable(requires_grad))
        cg.call_function(2, True)
        cg.store(varname)
        tx.output.pregraph_bytecode.extend(cg.get_instructions())

        # add the newly constructed nn.Parameter as a graph input
        source = LocalSource(varname)
        example_value = torch.nn.Parameter(
            tx.output.example_value_from_input_node(data.as_proxy().node)
        )
        result = VariableBuilder(tx, source)(example_value)
        # No need to guard on this since we already guarded on `data`.
        # These guards would fail since varname doesn't exist until after the function starts
        TracingContext.get().guards_context.dynamo_guards.remove_guards_with_source(
            source
        )
        return result
