import collections
import inspect
import logging

import math
import re
import types
from typing import Dict, List, Optional

import torch._C
import torch.fx
import torch.nn
import torch.onnx.operators
from torch._dynamo.utils import get_fake_value, get_real_value, torch_np
from torch._dynamo.variables import SymNodeVariable
from torch._dynamo.variables.user_defined import ProcessGroupVariable
from torch._guards import GuardsCheckpointState, Source
from torch.utils import _pytree as pytree

from .. import config, variables
from ..allowed_functions import torch_get_name
from ..exc import ArgsMismatchError, unimplemented, UserError, UserErrorType
from ..source import GeneratorStateSource, GetItemSource, NNModuleSource
from ..utils import (
    check_constant_args,
    check_unspec_python_args,
    deepcopy_to_fake_tensor,
    HAS_NUMPY,
    istype,
    np,
    product,
    proxy_args_kwargs,
    specialize_args_kwargs,
    tensortype_to_dtype,
)
from .base import VariableTracker
from .ctx_manager import AutocastModeVariable, NullContextVariable
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
    torch.get_default_dtype,
    torch.iinfo,
    torch.is_autocast_cache_enabled,
    torch.is_autocast_cpu_enabled,
    torch.is_autocast_enabled,
    torch.is_floating_point,
    torch.nn.functional._Reduction.get_enum,
]

constant_processgroup_functions = []

if torch.distributed.is_available():
    constant_fold_functions.append(torch.distributed.is_initialized)

    from torch.distributed.distributed_c10d import (
        _get_group_tag,
        get_process_group_ranks,
    )

    constant_processgroup_functions.extend(
        [
            get_process_group_ranks,
            _get_group_tag,
        ]
    )


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
                        torch_np._helpers.ndarrays_to_tensors,
                        *proxy_args_kwargs(args, {}),
                    ),
                    example_value=None,
                    **options,
                )
            else:
                unimplemented(f"torch.from_numpy(<{type(t)}>)")
        elif not config.dynamic_shapes and self.is_dynamic_shapes(args, kwargs):
            unimplemented(f"dynamic shapes: {self.value.__name__}")
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
            log.warning("Profiler will be ignored")
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
        elif (
            inspect.isfunction(self.value)
            and self.value in constant_processgroup_functions
        ):
            # becuase the input is a "ProcessGroupVariable", we'll be guarding on its
            # ID_MATCH based on how it was constructed.

            # We desugar it at trace-time into ranks by directly calling util
            # bake the result into the trace
            assert len(args) == 1, "Expected one arg (pg)"
            assert isinstance(args[0], ProcessGroupVariable)
            return ConstantVariable(self.value(args[0].as_python_constant()))
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

    def is_dynamic_shapes(self, args, kwargs):
        """Check for dynamic shapes when shape specialization is enabled"""
        # TODO(jansel): need to get a complete list
        if self.value in (
            torch.nonzero,
            torch.unique,
            torch.unique_consecutive,
        ) or self.value.__name__ in ("nms",):
            return True

        if self.value is torch.where and len(args) + len(kwargs) == 1:
            return True

        if self.value in (
            torch.arange,
            torch.repeat_interleave,
        ):
            none = variables.ConstantVariable(None)

            def has_non_const(it):
                return not all(x.is_python_constant() for x in it)

            def arange(start=none, end=none, step=none, **kwargs):
                return has_non_const([start, end, step])

            def repeat_interleave(input, repeats, dim=none, **kwargs):
                return has_non_const([repeats])

            return locals()[self.value.__name__](*args, **kwargs)

        return False

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


def safe_or_raise_always_restore(tx, graph_checkpoint, checkpoint, f, sub_args):
    # Will raise if not sound
    try:
        f.call_function(tx, sub_args, {})
    finally:
        tx.output.graph = graph_checkpoint
        tx.restore_graphstate(checkpoint)


# See NOTE [HigherOrderOperator tracing design] for details of the design
# See NOTE [speculate_subgraph vs old_speculate_subgraph] for other info
def speculate_subgraph(
    tx, f, sub_args, graph_checkpoint, checkpoint, *, always_restore=False
):
    from . import AutogradFunctionContextVariable, ConstantVariable, TensorVariable

    try:
        with tx.output.new_subtracer() as tracer:
            args = []
            # One argument to graph per sub_args
            for a in sub_args:
                if a is None:
                    a = ConstantVariable(None)
                assert not isinstance(
                    a, torch.Tensor
                ), "Tensors should already be tracked?"
                if isinstance(a, ConstantVariable):
                    proxy = tracer.create_graph_input("const")
                elif isinstance(a, (TensorVariable, AutogradFunctionContextVariable)):
                    tracer.create_graph_input(a.as_proxy().node.name)
                else:
                    raise unimplemented(
                        "HigherOrderOperator with body that accepts non-Tensors as input"
                    )
                args.append(a)
            output = f.call_function(tx, args, {})
            # Register output to graph
            # Modeled off of compile_and_call_fx_graph
            # TODO: support pytree output
            # We check always_restore because we dont use the output or side effects of always_restore code,
            # like bwd.
            if always_restore:
                # Nothing left to do here
                return output, tx.output.graph, tracer.lifted_freevars
            else:
                if not isinstance(
                    output, (TensorVariable, ListVariable, TupleVariable)
                ):
                    unimplemented("HigherOrderOperator with body with pytree output")

                if isinstance(output, (ListVariable, TupleVariable)):
                    if any(
                        not isinstance(var, TensorVariable)
                        for var in output.unpack_var_sequence(tx)
                    ):
                        unimplemented(
                            "HigherOrderOperator body's output must consist of tensors only"
                        )

                tx.output.guards.update(output.guards)
                tx.output.create_node(
                    "output",
                    "output",
                    (tracer.create_arg((output.as_proxy(),))),
                    {},
                )
                graph = tx.output.graph
                lifted_freevars = tracer.lifted_freevars

                return (
                    output,
                    graph,
                    lifted_freevars,
                )

    except torch._dynamo.exc.Unsupported as ex:
        log.warning(
            "TorchDynamo tracing of HigherOrderOperator did not go well. "
            "Falling back to eager behavior. This can result in a slowdown."
        )
        log.exception(ex)
        tx.output.graph = graph_checkpoint
        tx.restore_graphstate(checkpoint)
        raise


class TorchHigherOrderOperatorVariable(VariableTracker):
    def __init__(self, value, source: Optional[Source] = None, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.source = source

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import (
            ClosureVariable,
            ConstantVariable,
            ListVariable,
            NestedUserFunctionVariable,
            TensorVariable,
            UserFunctionVariable,
        )
        from .builder import wrap_fx_proxy

        assert (
            all(isinstance(value, ConstantVariable) for value in kwargs.values())
            or not kwargs
        ), "only constant kwargs are supported"

        def make_attr(name):
            node = tx.output.create_proxy(
                "get_attr",
                name,
                (),
                {},
            )
            return node

        def add_subgraph(name, gm):
            next_name = None
            i = 0
            while not next_name:
                candidate = f"{name}_{i}"
                if candidate in tx.output.nn_modules:
                    i += 1
                else:
                    next_name = candidate

            gm.__name__ = next_name
            src = NNModuleSource(GetItemSource(self.source, next_name))
            gm.torchdynamo_force_dynamic = False
            tx.output.register_attr_or_module(gm, next_name, source=src)
            return next_name

        def get_comparable_state(state):
            # Nub out bits of state that we don't require to be
            # equal
            return state._replace(
                output=state.output._replace(
                    guard_state=GuardsCheckpointState(set()),
                    nn_modules=None,
                    param_name_to_source=None,
                    # Timestamp is monotonically increasing so we don't
                    # care about divergence
                    timestamp=0,
                )
            )

        # NOTE: [speculate_subgraph vs old_speculate_subgraph]
        # We're in the middle of rewriting how Dynamo capture for HigherOrderOperators
        # works. If you are writing a new HigherOrderOperator, please prefer using
        # `speculate_subgraph`.
        # The main reason why we cannot get rid of `old_speculate_subgraph` yet is
        # that `speculate_subgraph` does not yet support calling nn.Modules or
        # accessing attributes of nn.Modules.
        def old_speculate_subgraph(f, sub_args, graph_checkpoint, checkpoint):
            if isinstance(f, NestedUserFunctionVariable) and f.closure is not None:
                # closure vars other than 'self' are not in scope of generated code, so error early
                # TODO(avik): we should eventually support this.
                # (Feature request tracked here: https://github.com/pytorch/pytorch/issues/99401)
                closure_vars = [
                    var.name
                    for var in f.closure.items
                    if isinstance(var, ClosureVariable) and var.name != "self"
                ]
                scope = {**tx.symbolic_locals, **tx.symbolic_globals}
                closure_vars = [
                    name
                    for name in closure_vars
                    if not isinstance(
                        scope[name], (UserFunctionVariable, NestedUserFunctionVariable)
                    )
                ]
                if closure_vars:
                    code = f.get_code()
                    raise torch._dynamo.exc.UserError(
                        torch._dynamo.exc.UserErrorType.ANTI_PATTERN,
                        f"Cannot create subgraph for nested function '{code.co_name}' "
                        f"at {code.co_filename}:{code.co_firstlineno} because "
                        f"it closes over variables {closure_vars}. Please rewrite "
                        f"'{code.co_name}' to take {closure_vars} as additional args.",
                        case_name="cond_closed_over_variable",
                    )

            # Setup the subgraph we're going to capture into
            tx.output.graph = torch.fx.Graph()
            tx.output.input_name_to_proxy.clear()

            args = []
            # One argument to graph per sub_args
            for a in sub_args:
                if isinstance(a, TensorVariable):
                    tx.output.current_tracer.create_graph_input(a.as_proxy().node.name)
                    args.append(a)
                else:
                    # call_function() needs a TensorVariable, therefore we construct
                    # one with inner graph proxy.
                    assert isinstance(a, torch.Tensor)
                    proxy = tx.output.current_tracer.create_graph_input("arg")
                    args.append(wrap_fx_proxy(tx=tx, proxy=proxy, example_value=a))
                # NB: we don't bother populating graphargs, as
                # they won't actually get used by anything

            output = f.call_function(tx, args, {})

            # Register output to graph
            # Modeled off of compile_and_call_fx_graph
            # TODO: support non single Tensor output
            if not isinstance(output, TensorVariable):
                raise ArgsMismatchError(
                    "Expected branch out type to be a single tensor but got {}".format(
                        str(output.python_type())
                    ),
                )
            tx.output.guards.update(output.guards)
            tx.output.create_node(
                "output",
                "output",
                (tx.output.current_tracer.create_arg((output.as_proxy(),))),
                {},
            )

            tx.output.side_effects.prune_dead_object_new(tx)
            state = tx.copy_graphstate()

            guards = state.output.guards
            nn_modules = state.output.nn_modules

            comparable_state = get_comparable_state(state)
            graph = tx.output.graph
            tx.output.graph = graph_checkpoint
            tx.restore_graphstate(checkpoint)

            return (
                output,
                graph,
                guards,
                nn_modules,
                comparable_state,
            )

        if self.value.__name__ == "cond":
            # TODO(voz): Support fake tensor dispatch for recursive
            # ops - see torch/dispatch/_dispatcher.py
            if len(args) != 4:
                raise UserError(
                    UserErrorType.DYNAMIC_CONTROL_FLOW,
                    f"Expected 4 arguments but got {len(args)}.\n"
                    f"Usage: cond(pred, true_fn, false_fn, operands)",
                )
            # predicate
            if type(args[0]) not in (ConstantVariable, TensorVariable, SymNodeVariable):
                raise UserError(
                    UserErrorType.DYNAMIC_CONTROL_FLOW,
                    f"Expected pred to be bool/int or a tensor with single "
                    f"item but got {str(type(args[0]))} "
                    f"with original python type {str(args[0].python_type())}.",
                )

            # operands
            if type(args[3]) is not ListVariable:
                raise UserError(
                    UserErrorType.DYNAMIC_CONTROL_FLOW,
                    f"Expected a list but got {args[3].python_type()}",
                )
            operands = args[3].unpack_var_sequence(tx)
            if not all(
                isinstance(operand, (TensorVariable, torch.Tensor))
                for operand in operands
            ):
                raise UserError(
                    UserErrorType.DYNAMIC_CONTROL_FLOW,
                    "Expected a list of tensors but got {actual_args}".format(
                        actual_args=[
                            str(operand.python_type())
                            if isinstance(operand, VariableTracker)
                            else str(type(operand))
                            for operand in operands
                        ],
                    ),
                )

            # branches
            assert isinstance(
                args[1], (UserFunctionVariable, NestedUserFunctionVariable)
            ), str(
                type(args[1])
            )  # true_fn
            assert isinstance(
                args[2], (UserFunctionVariable, NestedUserFunctionVariable)
            ), str(
                type(args[2])
            )  # false_fn

            # Our strategy for tracing the true/false branches of cond
            # are to checkpoint our graphstate, run the true branch,
            # roll it back to the checkpoint, and run the false
            # branch, and then merge the graphstates.  Well, perhaps
            # "merge" is too strong a word: we mostly assert that
            # the resulting graphstates have to be the same.
            #
            # We only permit guards to diverge (we union the guards from
            # both branches).  In particular, this means that side
            # effects are NOT permitted inside true/false branches; this
            # would be difficult to implement, because of the path
            # explosion problem.

            graph_checkpoint, checkpoint = tx.output.graph, tx.copy_graphstate()

            def speculate_branch(branch):
                try:
                    # NB: 0 is predicate
                    ix = 1 if branch else 2
                    return old_speculate_subgraph(
                        args[ix], operands, graph_checkpoint, checkpoint
                    )
                except ArgsMismatchError as e:
                    raise UserError(UserErrorType.DYNAMIC_CONTROL_FLOW, str(e))

            (
                true_r,
                true_graph,
                true_guards,
                true_nn_modules_context,
                true_cmp,
            ) = speculate_branch(True)
            (
                false_r,
                false_graph,
                false_guards,
                false_nn_modules_context,
                false_cmp,
            ) = speculate_branch(False)

            true_tracked_fakes = true_cmp.output.tracked_fakes
            false_tracked_fakes = false_cmp.output.tracked_fakes
            tx.output.tracked_fakes = list({*false_tracked_fakes, *true_tracked_fakes})
            true_tensor_weakref_to_sizes_strides = (
                true_cmp.output.tensor_weakref_to_sizes_strides
            )
            false_tensor_weakref_to_sizes_strides = (
                false_cmp.output.tensor_weakref_to_sizes_strides
            )

            # Add guards
            tx.output.tracing_context.guards_context.dynamo_guards |= false_guards
            tx.output.tracing_context.guards_context.dynamo_guards |= true_guards

            # Add tracking
            tx.output.tensor_weakref_to_sizes_strides.update(
                true_tensor_weakref_to_sizes_strides
            )
            tx.output.tensor_weakref_to_sizes_strides.update(
                false_tensor_weakref_to_sizes_strides
            )

            true_name = add_subgraph(
                "cond_true",
                torch.fx.GraphModule(true_nn_modules_context.nn_modules, true_graph),
            )
            false_name = add_subgraph(
                "cond_false",
                torch.fx.GraphModule(false_nn_modules_context.nn_modules, false_graph),
            )

            # Apply side effects (guaranteed to be equal)
            tx.output.side_effects = true_cmp.output.side_effects

            true_node = make_attr(true_name)
            false_node = make_attr(false_name)

            p_args = (
                args[0].as_proxy(),
                true_node,
                false_node,
                [a.as_proxy() for a in operands],
            )
            # TODO: assert that the true/false return values are
            # consistent
            example_value = true_r.as_proxy().node.meta["example_value"]
        elif self.value.__name__ == "map":
            assert type(args[0]) in (UserFunctionVariable, NestedUserFunctionVariable)
            assert type(args[1]) is TensorVariable

            sample_shape = args[1].get_real_value().size()
            if len(sample_shape) < 1 or sample_shape[0] == 0:
                unimplemented(
                    "map() operator doesn't support scalar or zero-sized tensors during tracing."
                )

            checkpoint = tx.copy_graphstate()
            # To get the example output from map() we will need to prodive at least one sample to
            # the loop body. In our case we will always use xs[0], and our map() won't support zero
            # sized tensor during tracing.
            (
                body_r,
                body_graph,
                body_guards,
                body_nn_modules_context,
                body_cmp,
            ) = old_speculate_subgraph(
                args[0],
                [
                    get_fake_value(args[1].as_proxy().node, tx)[0],
                    *args[2:],
                ],
                tx.output.graph,
                checkpoint,
            )

            # We don't support side effects inside a map loop body for simplicity.
            parent_cmp = get_comparable_state(checkpoint)
            parent_tracked_fakes = parent_cmp.output.tracked_fakes
            body_tracked_fakes = body_cmp.output.tracked_fakes
            tx.output.tracked_fakes = list({*parent_tracked_fakes, *body_tracked_fakes})
            body_tensor_weakref_to_sizes_strides = (
                body_cmp.output.tensor_weakref_to_sizes_strides
            )

            # Add guards
            tx.output.tracing_context.guards_context.dynamo_guards |= body_guards
            # Add tracking
            tx.output.tensor_weakref_to_sizes_strides.update(
                body_tensor_weakref_to_sizes_strides
            )

            body_name = add_subgraph(
                "map_body",
                torch.fx.GraphModule(body_nn_modules_context.nn_modules, body_graph),
            )

            body_node = make_attr(body_name)
            p_args = (body_node, *(arg.as_proxy() for arg in args[1:]))
            r = body_r.as_proxy().node.meta["example_value"]
            example_value = r.new_empty(
                [get_fake_value(args[1].as_proxy().node, tx).shape[0], *r.shape]
            )
        elif self.value.__name__ == "executorch_call_delegate":
            # This is operator for delegation within Executorch which calls a
            # specific function in the given lowered module with the given
            # operators. The actual operator is defined in the Executorch codebase.
            # This is a bad hierarchical violation since
            # executorch_call_delegate sits at a higher level than dynamo, but
            # there's no real solution to this issue yet.
            lowered_module = tx.output.get_submodule(args[0].module_key)

            lowered_node = make_attr(args[0].module_key)

            p_args = tuple(arg.as_proxy() for arg in args[1:])
            real_sub_args = pytree.tree_map_only(
                torch.fx.Proxy, lambda a: get_real_value(a.node, tx.output), p_args
            )
            example_res = lowered_module.original_module(*real_sub_args)
            example_value = deepcopy_to_fake_tensor(example_res, tx.fake_mode)

            p_args = (lowered_node,) + p_args
        elif self.value.__name__ in ("wrap", "wrap_activation_checkpoint"):
            # See NOTE [HigherOrderOperator tracing design] for more details
            checkpoint = tx.copy_graphstate()
            graph_checkpoint = tx.output.graph
            (
                body_r,
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                args[0],
                [
                    *args[1:],
                ],
                graph_checkpoint,
                checkpoint,
            )

            body_name = add_subgraph(
                "wrap_body", torch.fx.GraphModule(tx.output.nn_modules, body_graph)
            )
            body_node = make_attr(body_name)
            p_args = (
                body_node,
                *(arg.as_proxy() for arg in args[1:]),
                *(arg for arg in body_lifted_freevars.keys()),
            )
            example_value = pytree.tree_map_only(
                torch.fx.Proxy,
                lambda a: a.node.meta["example_value"],
                body_r.as_proxy(),
            )
        elif self.value.__name__ in (
            "trampoline_autograd_fwd",
            "trampoline_autograd_bwd",
            "trampoline_autograd_apply",
        ):
            from . import AutogradFunctionVariable, UserFunctionVariable

            pre_side_effects = tx.output.side_effects.clone()
            always_restore = self.value.__name__ == "trampoline_autograd_bwd"
            if (
                self.value.__name__ == "trampoline_autograd_bwd"
                or self.value.__name__ == "trampoline_autograd_fwd"
            ):
                fn = UserFunctionVariable(self.value, source=self.source)
            else:
                fn = TorchVariable(self.value)
            checkpoint = tx.copy_graphstate()
            pre_guards = tx.output.guards
            graph_checkpoint = tx.output.graph
            (
                body_r,
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                fn,
                [
                    *args,
                ],
                graph_checkpoint,
                checkpoint,
                # Backwards should never, ever be stored!
                always_restore=always_restore,
            )
            post_guards = tx.output.guards
            if body_lifted_freevars:
                for freevar in body_lifted_freevars.keys():
                    if "saved_tensor_marked" not in freevar.node.meta:
                        unimplemented("NYI - freevars in autograd function.")

            post_side_effects = tx.output.side_effects
            if post_side_effects.diff(pre_side_effects):
                diff = (
                    post_side_effects.id_to_variable.keys()
                    - pre_side_effects.id_to_variable.keys()
                )
                for d in diff:
                    if not isinstance(
                        post_side_effects.id_to_variable[d].value,
                        AutogradFunctionVariable,
                    ):
                        unimplemented("NYI - side effects in autograd function.")

            if always_restore:
                if post_guards - pre_guards:
                    unimplemented("NYI - New guards discovered in a restoring state")
                # Nothing left to do here
                return None

            p_args = (
                *(arg.as_proxy() for arg in args),
                *(arg for arg in body_lifted_freevars.keys()),
            )
            r = body_r.as_proxy().node.meta["example_value"]
            example_value = r
        else:
            unimplemented(f"HigherOrderOperator {self.value.__name__}")

        _, p_kwargs = proxy_args_kwargs([], kwargs)

        # Store the invocation as a call
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs=p_kwargs,
            ),
            example_value=example_value,
        )
