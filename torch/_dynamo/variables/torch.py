import collections
import logging

import math
import re
import types
from typing import Dict, List

import torch._C
import torch.fx
import torch.nn
import torch.onnx.operators
from torch._dynamo.variables import UserFunctionVariable

from .. import config, variables
from ..allowed_functions import torch_get_name
from ..exc import unimplemented
from ..source import GeneratorStateSource
from ..utils import (
    check_constant_args,
    check_unspec_python_args,
    HAS_NUMPY,
    istype,
    np,
    product,
    proxy_args_kwargs,
    specialize_args_kwargs,
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
from .tensor import TensorWithTFOverrideVariable

log = logging.getLogger(__name__)

# TODO(voz): Maybe rename these later
tensor_dunder_fns = [
    torch.Tensor.__rmatmul__,
    torch.Tensor.__rmod__,
    torch.Tensor.__rpow__,
    torch.Tensor.__rsub__,
    torch._C._TensorBase.__radd__,
    torch._C._TensorBase.__rmul__,
    torch._C._TensorBase.__ror__,
    torch._C._TensorBase.__rxor__,
    torch._C._TensorBase.__rand__,
]

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
    torch._C._get_privateuse1_backend_name,
]


if torch.distributed.is_available():
    constant_fold_functions.append(torch.distributed.is_initialized)


# TODO(voz): perhaps a decorator? This is rather readable for now tho, and not a public API.
def remap_as_fn___radd__(*args):
    return torch._C._TensorBase.__radd__(*args)


def remap_as_fn___rmul__(*args):
    return torch._C._TensorBase.__rmul__(*args)


def remap_as_fn___ror__(*args):
    return torch._C._TensorBase.__ror__(*args)


def remap_as_fn___rxor__(*args):
    return torch._C._TensorBase.__rxor__(*args)


def remap_as_fn___rand__(*args):
    return torch._C._TensorBase.__rand__(*args)


tensor_dunder_fns_remap = {
    torch._C._TensorBase.__radd__: remap_as_fn___radd__,
    torch._C._TensorBase.__rmul__: remap_as_fn___rmul__,
    torch._C._TensorBase.__ror__: remap_as_fn___ror__,
    torch._C._TensorBase.__rxor__: remap_as_fn___rxor__,
    torch._C._TensorBase.__rand__: remap_as_fn___rand__,
}


try:
    # Wed need to monkeypatch transformers here, sadly.
    # TODO(voz): Upstream to transformers lib
    import transformers

    def _dynamo_overriden_transformers_eq(self, other):
        if not hasattr(other, "__dict__"):
            return False
        return self.__dict__ == other.__dict__

    transformers.configuration_utils.PretrainedConfig.__eq__ = (
        _dynamo_overriden_transformers_eq
    )
except ImportError:
    pass


class TorchVariable(VariableTracker):
    """Points to a module or method in torch.*"""

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        if (
            isinstance(value, collections.abc.Hashable)
            and value in tensor_dunder_fns_remap
        ):
            value = tensor_dunder_fns_remap[value]

        self.value = value

        # the remainder of this is just optional debug checks
        try:
            self_should_be_none = getattr(self.value, "__self__", None)
        except RuntimeError as e:
            assert "No such operator" in str(e), str(e)
            self_should_be_none = None

        # assert "_ntuple.<locals>.parse" not in str(value)

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

    def call_hasattr(self, tx, name):
        result = hasattr(self.value, name)
        return variables.ConstantVariable(result).add_options(self)

    def unique_var_name(self):
        name = torch_get_name(self.value, f"allowed_fn_{id(self.value)}")
        return "__" + re.sub(r"[^a-zA-Z0-9_]+", "_", name)

    def reconstruct(self, codegen):
        return codegen.setup_globally_cached(self.unique_var_name(), self.value, False)

    def as_proxy(self):
        return self.value

    def python_type(self):
        if isinstance(self.value, (torch.Tensor, torch.nn.Module)):
            return type(self.value)
        if isinstance(self.value, type):
            return type
        return super().python_type()

    def as_python_constant(self):
        return self.value

    def can_constant_fold_through(self):
        if self.value in constant_fold_functions:
            return True
        return getattr(self.value, "__module__", None) == "math"

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import (
            ConstantVariable,
            CUDAStreamContextVariable,
            CUDAStreamVariable,
            DeterministicAlgorithmsVariable,
            DisabledSavedTensorsHooksVariable,
            GradModeVariable,
            SymNodeVariable,
            TensorVariable,
            UserDefinedObjectVariable,
        )

        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls

        constant_args = check_constant_args(args, kwargs)
        unspec_python_args = check_unspec_python_args(args, kwargs)
        options = VariableTracker.propagate(self, args, kwargs.values())

        if self.value in config.constant_functions:
            assert not args and not kwargs
            return ConstantVariable(config.constant_functions[self.value], **options)
        elif self.value is torch._functorch.eager_transforms.grad_impl:
            op = TorchHigherOrderOperatorVariable.make(
                self.value,
                source=self.source,
            ).call_function(tx, args, kwargs)
            return op
        elif self.can_constant_fold_through() and (constant_args or unspec_python_args):
            args, kwargs = specialize_args_kwargs(tx, args, kwargs)
            # constant fold
            return ConstantVariable(
                self.as_python_constant()(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
                **options,
            )
        elif istype(self.value, type) and issubclass(self.value, torch.nn.Module):
            if self.value is torch.nn.CrossEntropyLoss:
                return self._call_cross_entropy_loss(tx, args, kwargs, options)
            else:
                return variables.UserDefinedClassVariable(
                    self.value, source=self.source, **options
                ).call_function(tx, args, kwargs)
        elif self.value in (torch.is_tensor, torch.overrides.is_tensor_like):
            assert len(args) == 1
            if isinstance(args[0], TensorVariable) or (
                self.value is torch.overrides.is_tensor_like
                and isinstance(args[0], UserDefinedObjectVariable)
                and hasattr(args[0].value, "__torch_function__")
            ):
                return ConstantVariable(True, **options)
            else:
                return ConstantVariable(False, **options)
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
                    return ConstantVariable(
                        input_arg.dtype.is_floating_point, **options
                    )
                elif self.value is torch.is_complex:
                    return ConstantVariable(input_arg.dtype.is_complex, **options)
                else:
                    raise AssertionError(f"calling {self.value}")
        elif (
            self.value is torch.numel
            and isinstance(args[0], TensorVariable)
            and args[0].size is not None
        ):
            return ConstantVariable(product(args[0].size), **options)
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
            return self._call_ntuple(tx, args, kwargs, options)
        elif self.value is torch.no_grad:
            return GradModeVariable.create(tx, False, **options)
        elif self.value is torch.enable_grad:
            return GradModeVariable.create(tx, True, **options)
        elif self.value is torch.set_grad_enabled and len(args) == 1:
            return GradModeVariable.create(tx, args[0].as_python_constant(), **options)
        elif self.value is torch.is_grad_enabled:
            assert not (args or kwargs)
            return ConstantVariable(torch.is_grad_enabled(), **options).add_guards(
                GradModeVariable._guards_singleton
            )
        elif self.value is torch.use_deterministic_algorithms and len(args) == 1:
            return DeterministicAlgorithmsVariable.create(
                tx, args[0].as_python_constant(), **options
            )
        elif self.value is torch.are_deterministic_algorithms_enabled:
            assert not (args or kwargs)
            return ConstantVariable(
                torch.are_deterministic_algorithms_enabled(), **options
            ).add_guards(DeterministicAlgorithmsVariable._guards_singleton)
        elif self.value is torch.autograd.graph.disable_saved_tensors_hooks:
            assert len(args) == 1
            return DisabledSavedTensorsHooksVariable.create(
                tx, args[0].as_python_constant(), **options
            )
        elif self.value is torch._C._is_torch_function_enabled:
            assert not (args or kwargs)
            return ConstantVariable(
                tx.output.torch_function_enabled, **options
            ).add_guards(TorchFunctionDisableVariable._guards_singleton)
        elif self.value is torch._C.DisableTorchFunctionSubclass:
            assert not (args or kwargs)
            return TorchFunctionDisableVariable.create(tx, **options)
        elif self.value is torch.cuda.stream:
            log.warning(
                "torch.cuda.stream() not fully supported, streams may be ignored"
            )
            assert len(args) == 1
            return CUDAStreamContextVariable.create(tx, args[0], **options)
        elif self.value is torch.cuda.streams.Stream:
            return wrap_fx_proxy_cls(
                CUDAStreamVariable,
                tx,
                tx.output.create_proxy(
                    "call_function",
                    torch.cuda.streams.Stream,
                    (),
                    {},
                ),
                **options,
            )
        elif self.value is torch.from_numpy:
            if not config.numpy_ndarray_as_tensor:
                unimplemented(
                    "torch.from_numpy(). Turn on config.numpy_ndarray_as_tensor to support "
                    "torch.from_numpy()."
                )
            assert len(args) == 1, f"Got arguments {args}"
            assert not kwargs
            t = args[0]
            from .tensor import NumpyNdarrayVariable

            if isinstance(t, NumpyNdarrayVariable):
                # TODO: mark the tensor as non-resizable
                return wrap_fx_proxy_cls(
                    target_cls=TensorVariable,
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        torch.detach,
                        *proxy_args_kwargs(args, {}),
                    ),
                    example_value=None,
                    **options,
                )
            else:
                unimplemented(f"torch.from_numpy(<{type(t)}>)")
        elif len(args) > 0 and isinstance(args[0], TensorWithTFOverrideVariable):
            # This code block implements inlining the __torch_function__
            # override of a tensor.

            tensor_with_tf_override = args[0]

            # TODO(future PR): make this implement the full __torch_function__ API
            # instead of assuming the relevant override is in the first argument.
            args[0] = args[0].tensor_variable

            unwrapped = TensorWithTFOverrideVariable.inline_torch_function_unwrapped(
                tx,
                self,
                tensor_with_tf_override.orig_tensor_variable_source,
                tensor_with_tf_override.subclass_torch_function__func,
                tensor_with_tf_override.subclass_type,
                options,
                args,
                kwargs,
            )

            # The wrapping here follows the logic in
            # `torch.Tensor.__torch_function__`.
            if self.value in torch.overrides.get_default_nowrap_functions():
                return unwrapped
            return TensorWithTFOverrideVariable(
                unwrapped,
                tensor_with_tf_override.orig_tensor_variable_source,
                tensor_with_tf_override.subclass_torch_function__func,
                tensor_with_tf_override.subclass_type,
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
            return NullContextVariable(**options)
        elif self.value is torch.autograd._profiler_enabled:
            unimplemented("torch.autograd._profiler_enabled not supported yet")
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
            return ConstantVariable(
                torch.backends.cudnn.is_acceptable(tensor_inp), **options
            )
        elif self.value is torch.nn.Parameter:
            # https://github.com/pytorch/pytorch/issues/99569
            unimplemented("torch.nn.Parameter not supported")
        if (
            self.value.__name__ == "get_state"
            and hasattr(self.value, "__self__")
            and isinstance(self.value.__self__, torch._C.Generator)
        ):

            def get_state_from_generator():
                return self.value()

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    get_state_from_generator,
                    *proxy_args_kwargs(args, kwargs),
                ),
                example_value=self.value(),
                source=GeneratorStateSource(
                    self.value.__self__.device.type, self.value.__self__.initial_seed()
                ),
                **options,
            )
        if (
            self.value.__name__ == "set_state"
            and hasattr(self.value, "__self__")
            and isinstance(self.value.__self__, torch._C.Generator)
        ) or self.value == torch.random.set_rng_state:
            assert len(args) == 1
            assert isinstance(args[0], TensorVariable)

            unimplemented(
                "TODO: make torch.random.set_rng_state work with FakeTensor/aot_autograd"
            )
            # In fake tensor case, this state doesn't matter, but
            # it needs to be valid to not segfault. Pull a real tensor out.
            # The value won't matter since we are running with fake tensors anyway, so rng doesn't matter.
            # However, it is imperative to record the call_function in the graph with the true args
            # (Not the fake example_value) - for the sake of graph correctness.
            if self.value == torch.random.set_rng_state:
                example_value = torch.random.get_rng_state()
            else:
                example_value = self.value.__self__.get_state()

            self.value.__module__ = self.__module__
            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    *proxy_args_kwargs(args, kwargs),
                ),
                example_value=example_value,
                **options,
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
                **options,
            )
        elif (
            self.value == torch.addcdiv
            and len(args) == 3
            and "value" in kwargs
            and len(kwargs) == 1
        ):
            # decompose addcdiv into constituent ops, prevents a graph break due to converting
            # value to a scalar
            result = TorchVariable(torch.div, **options).call_function(tx, args[1:], {})
            result = TorchVariable(torch.mul, **options).call_function(
                tx, [result, kwargs["value"]], {}
            )
            return TorchVariable(torch.add, **options).call_function(
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
            # guard propagaiton via options is the best we can do.
            from .builder import SourcelessBuilder

            return SourcelessBuilder()(tx, invocation_result).add_options(options)
        elif is_from_local(self.value):
            # rewrite non-primitive args/kwargs to be included in the on-the-fly prim function
            # and rewrite args to have only proxyable args, then insert call_function
            args_as_value = [x.as_python_constant() for x in args[1:]]

            def fn_with_prim_types(x, **kwargs):
                return self.value(x, *args_as_value, **kwargs)

            # attach the same function name for better debugging
            fn_with_prim_types.__name__ = "prim " + self.value.__name__

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    fn_with_prim_types,
                    *proxy_args_kwargs([args[0]], kwargs),
                ),
                **options,
            )
        elif self.value == torch.nn.init._calculate_correct_fan:
            return UserFunctionVariable(
                torch.nn.init._calculate_correct_fan, **options
            ).call_function(tx, args, {})
        elif self.value == torch.utils._pytree.tree_flatten:
            if len(args) != 1:
                unimplemented("Unsupported flatten with len(args) != 1")

            flattened, spec = torch.utils._pytree.tree_flatten(args[0])
            return TupleVariable(
                [ListVariable(flattened), ConstantVariable(spec)], **options
            )
        elif self.value == torch.utils._pytree.tree_unflatten:
            if len(args) != 2:
                unimplemented("Unsupported unflatten with len(args) != 2")

            return torch.utils._pytree.tree_unflatten(args[0], args[1].value)
        elif self.value == torch.utils._pytree.tree_map_only:
            if len(args) != 3:
                unimplemented("Unsupported tree_map_only with len(args) != 3")

            ty = args[0].value  # type
            fn = args[1]  # map fn
            tree = args[2]  # tree

            def map_fn(v):
                if ty == v.python_type():
                    return fn.call_function(tx, [v], {})
                else:
                    return v

            return torch.utils._pytree.tree_map(map_fn, tree)
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
            # Handle sth like torch.LongTensor(list(np.int64, np.int64, ...)),
            # as FX symbolic trace doesn't support numpy int/float as base types.
            if (
                HAS_NUMPY
                and self.value in tensortype_to_dtype
                and len(args) == 1
                and isinstance(args[0], ListVariable)
                and args[0].is_python_constant()
            ):
                for x in args[0].items:
                    if isinstance(x.value, np.generic):
                        x.value = x.value.item()

            # TODO(voz): Replace w/ dynamic shape rewrite table.
            # Ideally, we would be able to do this at ctor time, but alas we need a combination
            # of value + args to determine this.
            fn_ = self.value
            if any(isinstance(x, SymNodeVariable) for x in args):
                if self.value == math.sqrt:
                    from torch.fx.experimental.symbolic_shapes import sym_sqrt

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

                # NB: OK to pass torch.tensor(tensor), this will trace fine
                # TODO: But torch.tensor(unspec) would not trace fine.  Not
                # handled right now.
                data_arg = None
                if args:
                    data_arg = args[0]
                elif "data" in kwargs:
                    data_arg = kwargs["data"]

                if isinstance(data_arg, ListVariable) and check_any_unspec(data_arg):
                    unimplemented("torch.tensor call with list of unspec")
            tensor_variable = wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    fn_,
                    *proxy_args_kwargs(args, kwargs),
                ),
                **options,
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
                    name = tx.find_symbolic_locals_name(kwargs["out"])
                    if name in tx.symbolic_locals:
                        tx.symbolic_locals[name] = tensor_variable
                else:
                    unimplemented(f"out variant of {type(kwargs['out'])}")

            return tensor_variable

    def _call_cross_entropy_loss(self, tx, args, kwargs, options):
        """
        functional: input, target, weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional ctor: weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional loss call: input, target, optional_output
        """
        from . import ConstantVariable

        def normalize_args(
            weight=ConstantVariable(None),
            size_average=ConstantVariable(None),
            ignore_index=ConstantVariable(-100),
            reduce=ConstantVariable(None),
            reduction=ConstantVariable("mean"),
            label_smoothing=ConstantVariable(0.0),
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
                **VariableTracker.propagate(
                    [
                        self,
                        weight,
                        size_average,
                        ignore_index,
                        reduce_arg,
                        reduction,
                        label_smoothing,
                        input,
                        target,
                    ]
                ),
            )

        return variables.LambdaVariable(fake_cross_entropy_loss, **options)

    def _call_ntuple(self, tx, args, kwargs, options):
        """inline behavior of torch.nn.modules.utils._ntuple"""
        if self.value is torch.nn.modules.utils._ntuple:
            count = args[0].as_python_constant()
        else:
            count = self.value.__closure__[0].cell_contents
        assert isinstance(count, int)

        def handle_ntuple(value):
            if value.has_unpack_var_sequence(tx):
                return variables.TupleVariable(
                    list(value.unpack_var_sequence(tx)),
                    **VariableTracker.propagate(self, value, args, kwargs.values()),
                )
            elif value.is_python_constant():
                # constant prop through it
                return variables.ConstantVariable(
                    torch.nn.modules.utils._ntuple(count)(value.as_python_constant()),
                    **VariableTracker.propagate(self, value, args, kwargs.values()),
                )
            else:
                unimplemented(f"torch.nn.modules.utils._ntuple({value})")

        if self.value is torch.nn.modules.utils._ntuple:
            return variables.LambdaVariable(handle_ntuple, **options)
        else:
            return handle_ntuple(args[0])
