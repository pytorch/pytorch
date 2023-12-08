import inspect
import logging

import math
import re
import types
from typing import Dict, List

from torch._streambase import _StreamBase
from ..guards import install_guard

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

import torch._C
import torch._refs
import torch.fx
import torch.nn
import torch.onnx.operators

from .. import config, polyfill, variables
from ..allowed_functions import torch_get_name
from ..device_interface import get_registered_device_interfaces
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..utils import (
    check_constant_args,
    check_unspec_python_args,
    has_torch_function,
    istype,
    product,
    proxy_args_kwargs,
    tensortype_to_dtype,
)
from .base import VariableTracker
from .ctx_manager import (
    AutocastModeVariable,
    NullContextVariable,
    TorchFunctionDisableVariable,
)
from .distributed import is_constant_pg_functions, is_from_local, ProcessGroupVariable
from .higher_order_ops import TorchHigherOrderOperatorVariable
from .lists import ListVariable, TupleVariable
from .torch_function import can_dispatch_torch_function, dispatch_torch_function

log = logging.getLogger(__name__)


torch_special_class_types = (torch._C.Generator,)

REWRITE_OPS_TO_TENSOR_SIZE_METHOD = [
    torch.onnx.operators.shape_as_tensor,
    torch._shape_as_tensor,
]

constant_fold_functions = [
    torch._assert,
    torch._utils._get_device_index,
    torch.cuda.is_available,
    torch.device,
    torch.distributed.is_available,
    torch.finfo,
    torch.get_autocast_gpu_dtype,
    torch.get_default_dtype,
    torch.iinfo,
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
        name = torch_get_name(value, f"allowed_fn_{id(value)}")
        unique_var_name = "__" + re.sub(r"[^a-zA-Z0-9_]+", "_", name)
        return codegen.setup_globally_cached(unique_var_name, value, False)

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

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import GradModeVariable, InferenceModeVariable, StreamVariable

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
            return InferenceModeVariable.create(tx, args[0].as_python_constant())
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
        elif self.value in [
            torch.amp.autocast_mode.autocast,
            torch.cuda.amp.autocast,
            torch.cpu.amp.autocast,
        ]:
            return AutocastModeVariable.create(self.value, args, kwargs)
        elif self.value in (
            torch.profiler.profile,
            torch.profiler.record_function,
            torch.autograd.profiler.profile,
            torch.autograd.profiler.record_function,
        ):
            log.warning("Profiler function %s will be ignored", self.value)
            return NullContextVariable()
        elif self.value is torch._C.DisableTorchFunctionSubclass:
            assert not (args or kwargs)
            return TorchFunctionDisableVariable.create(tx)


class TorchInGraphFunctionVariable(BaseTorchVariable):
    """Points to a torch function/method that should be put in FX graph"""

    def __repr__(self):
        return f"TorchInGraphFunctionVariable({self.value})"

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import (
            ConstantVariable,
            DeterministicAlgorithmsVariable,
            DisabledSavedTensorsHooksVariable,
            GradModeVariable,
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
            if self.value in [
                torch._utils.is_compiling,
                torch._dynamo.external_utils.is_compiling,
            ]:
                tx.mark_inconsistent_side_effects()
            return ConstantVariable.create(tracing_state_functions[self.value])
        elif self.value in (
            torch._functorch.vmap.vmap_impl,
            torch._functorch.eager_transforms.grad_impl,
        ):
            return TorchHigherOrderOperatorVariable.make(
                self.value,
                source=self.source,
            ).call_function(tx, args, kwargs)
        elif self.value is torch.overrides.get_default_nowrap_functions:
            # [Note: __torch_function__] we return empty here because we restrict
            # the set of functions that we trace __torch_function__ on to
            # functions outside of the actual set. Implementing this properly will require implementing
            # some variable types to track and compare tensor getset descriptors
            from .builder import SourcelessBuilder

            return SourcelessBuilder()(
                tx, torch.overrides.get_default_nowrap_functions()
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
        elif self.value is torch.autograd.graph.disable_saved_tensors_hooks:
            assert len(args) == 1
            return DisabledSavedTensorsHooksVariable.create(
                tx, args[0].as_python_constant()
            )
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
            self.value in [torch.ops.aten.sym_size, torch.ops.aten.sym_size.int]
            and len(args) == 2
            and len(kwargs) == 0
            and isinstance(args[0], TensorVariable)
        ):
            # we see this when retracing already traced code
            return args[0].call_method(tx, "size", [args[1]], {})
        elif (
            self.value is [torch.ops.aten.sym_stride, torch.ops.aten.sym_stride.int]
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
        elif is_constant_pg_functions(self.value):
            # becuase the input is a "ProcessGroupVariable", we'll be guarding on its
            # ID_MATCH based on how it was constructed.

            # We desugar it at trace-time into ranks by directly calling util
            # bake the result into the trace
            assert len(args) == 1, "Expected one arg (pg)"
            assert isinstance(args[0], ProcessGroupVariable)

            invocation_result = self.value(args[0].as_python_constant())
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
                if self.value == math.sqrt:
                    from torch.fx.experimental.sym_node import sym_sqrt

                    fn_ = sym_sqrt

            if fn_ is torch.tensor:

                def check_any_unspec(x):
                    # NB: This includes UnspecializedPythonVariable
                    if isinstance(x, (TensorVariable, SymNodeVariable)):
                        return True
                    elif isinstance(x, ListVariable):
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
                elif isinstance(tensor_variable, TensorVariable):
                    assert isinstance(kwargs["out"], TensorVariable)
                    if (
                        kwargs["out"].source
                        and kwargs["out"] in tx.output.graphargs
                        and kwargs["out"].size != tensor_variable.size
                    ):
                        # It's hard to get out variants with resizing on graph inputs work
                        # properly across dynamo/aot/inductor, just fall back.
                        unimplemented("out variants with resizing on graph inputs")
                    assert "example_value" in kwargs["out"].proxy.node.meta
                    if not torch._prims_common.is_contiguous(
                        kwargs["out"].proxy.node.meta["example_value"]
                    ):
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


class TorchVariable(BaseTorchVariable):
    """Points to a module, classes or functions in torch.*"""

    def __init__(self, value, **kwargs):
        assert not isinstance(
            value, (torch.dtype, torch.device)
        ), "should use ConstantVariable"

        super().__init__(value, **kwargs)

        # the remainder of this is just optional debug checks
        try:
            self_should_be_none = getattr(self.value, "__self__", None)
        except RuntimeError as e:
            assert "No such operator" in str(e), str(e)
            self_should_be_none = None
        except AssertionError as e:
            assert "Unknown attribute" in str(e), str(e)
            self_should_be_none = None

        if self_should_be_none is None:
            pass
        elif isinstance(self_should_be_none, types.ModuleType):
            # weird ones like torch.nn.functional.avg_pool2d have __self__
            name = self_should_be_none.__name__
            assert re.match(r"^(torch|math)([.]|$)", name), f"__self__ set to {name}"
        elif isinstance(
            self_should_be_none, type(torch._C._get_tracing_state.__self__)
        ):
            # some _C functions have __self__ as a null capsule
            pass
        elif isinstance(self_should_be_none, torch_special_class_types):
            pass
        else:
            raise AssertionError(f"{value} found with __self__ set")

    def __repr__(self):
        return f"TorchVariable({self.value})"

    def python_type(self):
        if isinstance(self.value, (torch.Tensor, torch.nn.Module, torch.device)):
            return type(self.value)
        if isinstance(self.value, type):
            return type
        return super().python_type()

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import ConstantVariable

        from .builder import wrap_fx_proxy

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
        elif istype(self.value, type) and issubclass(self.value, torch.nn.Module):
            if self.value is torch.nn.CrossEntropyLoss:
                return self._call_cross_entropy_loss(tx, args, kwargs)
            else:
                return variables.UserDefinedClassVariable(
                    self.value, source=self.source
                ).call_function(tx, args, kwargs)
        elif can_dispatch_torch_function(tx, args, kwargs):
            return dispatch_torch_function(tx, self, args, kwargs)
        elif isinstance(self.value, types.ModuleType):
            unimplemented("TypeError(\"'module' object is not callable\")")
        else:
            # torch.LongTensor cannot accept a list of FakeTensors.
            # So we stack the list of FakeTensors instead.
            if (
                np
                and self.value in tensortype_to_dtype
                and len(args) == 1
                and isinstance(args[0], ListVariable)
                and len(args[0].items) > 1
                and all(isinstance(x, variables.TensorVariable) for x in args[0].items)
            ):
                # Stack FakeTensor
                stacked = wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        torch.stack,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                )
                args = [stacked]

            tensor_variable = wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    *proxy_args_kwargs(args, kwargs),
                ),
            )

            return tensor_variable

    def _call_cross_entropy_loss(self, tx, args, kwargs):
        """
        functional: input, target, weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional ctor: weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional loss call: input, target, optional_output
        """
        from . import ConstantVariable

        def normalize_args(
            weight=ConstantVariable.create(None),
            size_average=ConstantVariable.create(None),
            ignore_index=ConstantVariable.create(-100),
            reduce=ConstantVariable.create(None),
            reduction=ConstantVariable.create("mean"),
            label_smoothing=ConstantVariable.create(0.0),
        ):
            return (
                weight,
                size_average,
                ignore_index,
                reduce,
                reduction,
                label_smoothing,
            )

        (
            weight,
            size_average,
            ignore_index,
            reduce_arg,
            reduction,
            label_smoothing,
        ) = normalize_args(*args, **kwargs)

        def fake_cross_entropy_loss(input, target):
            from .builder import wrap_fx_proxy

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch.nn.functional.cross_entropy,
                    *proxy_args_kwargs(
                        [
                            input,
                            target,
                            weight,
                            size_average,
                            ignore_index,
                            reduce_arg,
                            reduction,
                            label_smoothing,
                        ],
                        {},
                    ),
                ),
            )

        return variables.LambdaVariable(fake_cross_entropy_loss)
