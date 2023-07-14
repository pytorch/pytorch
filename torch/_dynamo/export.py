import dataclasses
import inspect
import logging
import traceback
import types
import weakref
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from unittest.mock import patch

import sympy

import torch
import torch.utils._pytree as pytree
from torch import _guards
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses import fake_tensor
from torch.fx.experimental.proxy_tensor import make_fx

from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    StrictMinMaxConstraint,
)

from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRanges
from . import config, skipfiles
from .eval_frame import check_if_dynamo_supported, innermost_fn, optimize_assert
from .exc import CondOpArgsMismatchError, UserError, UserErrorType
from .hooks import Hooks

log = logging.getLogger(__name__)
if TYPE_CHECKING:
    from torch._C._dynamo.eval_frame import reset_code  # noqa: F401
else:
    for name in dir(torch._C._dynamo.eval_frame):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(torch._C._dynamo.eval_frame, name)


def remove_from_cache(f):
    """
    Make sure f.__code__ is not cached to force a recompile
    """
    if isinstance(f, types.CodeType):
        reset_code(f)
    elif hasattr(f, "__code__"):
        reset_code(f.__code__)
    elif hasattr(getattr(f, "forward", None), "__code__"):
        reset_code(f.forward.__code__)
    else:
        from . import reset

        reset()
        log.warning("could not determine __code__ for %s", f)


@dataclasses.dataclass
class ConstraintTarget:
    """
    This represents input tensor dimensions.  Don't create this
    class directly; instead, use :func:`torch._export.dynamic_dim`.
    """

    w_tensor: weakref
    # TODO: We don't need t_id; we can get it off of w_tensor
    t_id: int
    dim: int


@dataclasses.dataclass
class Constraint(ConstraintTarget):
    """
    This represents constraints on input tensor dimensions, e.g., requiring
    them to be fully polymorphic or within some range.  Don't create this
    class directly; instead, use :func:`torch._export.dynamic_dim`.
    """

    # NOTE(avik): In the future, this could be Union[StrictMinMaxConstraint, <other kinds>]
    constraint_range: StrictMinMaxConstraint
    # Represent that `constraint_range` is shared with another ConstraintTarget, which
    # typically arises because of a specified equality with another dynamic dimension.
    shared: Optional[ConstraintTarget] = None

    def _clone_with_range(self, lower=2, upper=sympy.oo):
        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & ValueRanges(lower=lower, upper=upper),
            warn_only=False,
        )
        return Constraint(
            self.w_tensor, self.t_id, self.dim, constraint_range, self.shared
        )

    def __ge__(self, lower):
        return self._clone_with_range(lower=lower)

    def __gt__(self, lower):
        return self._clone_with_range(lower=lower + 1)

    def __le__(self, upper):
        return self._clone_with_range(upper=upper)

    def __lt__(self, upper):
        return self._clone_with_range(upper=upper - 1)

    def __bool__(self):
        # NOTE(avik): We do not support compound expressions like a <= x <= b.
        # This is because Python implicitly desugars them into bool(a <= x) and bool(x <= b),
        # and moreover, enforces that any overload of __bool__ must return True or False.
        # FWIW, sympy also raises TypeError in this case.
        raise TypeError(
            "Cannot determine truth value of Constraint. "
            "If you are trying to combine Constraints with logical connectives, "
            "you can specify them separately instead."
        )

    @property
    def serializable_spec(self):
        # We need a serialization compatible format of the constraint so that it
        # can be savedin the graph module w/o breaking the module serialization.
        # The saved constraints will be used directly for the post-exporting pass
        # that converts constraints to runtime assertion. The saved constraints
        # will not be saved in the serialized module.
        # TODO: A better way is needed. Currently we use 't_id' to map the constraint,
        # which is not reliable
        return {
            "t_id": self.t_id,
            "dim": self.dim,
            "min": self.constraint_range.vr.lower,
            "max": self.constraint_range.vr.upper,
            "shared": (
                None
                if self.shared is None
                else {
                    "t_id": self.shared.t_id,
                    "dim": self.shared.dim,
                }
            ),
        }

    def __eq__(self, other):
        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & other.constraint_range.vr,
            warn_only=False,
        )
        return Constraint(
            self.w_tensor,
            self.t_id,
            self.dim,
            constraint_range,
            shared=ConstraintTarget(other.w_tensor, other.t_id, other.dim),
        )


class FlattenInputOutputSignature(torch.fx.interpreter.Transformer):
    def __init__(
        self,
        m: torch.fx.GraphModule,
        flat_args: Tuple[Any],
        matched_input_elements_positions: List[int],
        matched_output_elements_positions: List[int],
        example_fake_inputs: List[torch.Tensor],
        fake_mode: Optional[fake_tensor.FakeTensorMode] = None,
    ):
        super().__init__(m)

        matched_input_elements_to_fake = {
            val: example_fake_inputs[ix]
            for ix, val in enumerate(matched_input_elements_positions)
        }

        self.new_args = []
        for i in range(0, len(flat_args)):
            arg = super(FlattenInputOutputSignature, self).placeholder(
                f"arg{i}", (), {}
            )
            if i in matched_input_elements_to_fake:
                arg.node.meta["val"] = matched_input_elements_to_fake[i]
            else:
                # Fill node.mata["val"] with faketensor from the input,
                # if it's not found in matched_input_elements_positions
                if fake_mode is not None and isinstance(flat_args[i], torch.Tensor):
                    arg.node.meta["val"] = fake_mode.from_tensor(flat_args[i])
            self.new_args.append(arg)
        self.old_args_gen = (self.new_args[i] for i in matched_input_elements_positions)
        self.matched_output_elements_positions = matched_output_elements_positions

    def placeholder(self, target, args, kwargs):
        arg = next(self.old_args_gen)
        if "val" in self.current_node.meta:
            arg.node.meta["val"] = self.current_node.meta["val"]
        if "tensor_dict" in self.current_node.meta:
            arg.node.meta["tensor_dict"] = self.current_node.meta["tensor_dict"]
        return arg

    def output(self, target, args, kwargs):
        dynamo_result_flat = args[0]
        lookup = [*dynamo_result_flat, *self.new_args]
        new_result_flat = [lookup[i] for i in self.matched_output_elements_positions]
        return super().output(target, (new_result_flat,), {})

    def run_node(self, n):
        self.current_node = n
        r = super().run_node(n)
        if "val" in self.current_node.meta:
            r.node.meta["val"] = self.current_node.meta["val"]
        return r


def export(
    f: Callable[..., Any],
    *args,
    aten_graph: bool = False,
    pre_dispatch: bool = False,
    decomposition_table: Optional[
        Dict[torch._ops.OpOverload, Callable[..., Any]]
    ] = None,
    tracing_mode: str = "symbolic",
    constraints: Optional[List[Constraint]] = None,
    assume_static_by_default: bool = False,
    fake_mode: fake_tensor.FakeTensorMode = None,
    **kwargs,
) -> Tuple[torch.fx.GraphModule, Set[_guards.Guard]]:
    """
    Export an input function f to a format that can be executed outside of PyTorch using the FX graph.

    Args:
        f (callable): A PyTorch function to be exported.

        *args: Variable length argument list to be passed to the function f.

        aten_graph (bool): If True, exports a graph with ATen operators.
        If False, exports a graph with Python operators. Default is False.

        pre_dispatch (bool): If True, exports a graph with ATen operators,
        but before any logic in the PyTorch dispatcher has run.
        This can be useful if you want to apply further tranformations on a graph before running it
        through autograd, autocast, or any other functionalities that are integrated into the dispatcher.
        This flag is only valid if aten_graph=True is set.
        Default is False.

        decomposition_table (dict): A dictionary that maps operators to their decomposition functions.
        Required if aten_graph or tracing_mode is specified. Default is None.

        tracing_mode (str): If "symbolic", turn on dynamic shapes support. Default is "symbolic".

        fake_mode (fake_tensor.FakeTensorMode): Use this fake_mode instead of creating an internal one.
        Useful during symbolic tracing, when user input is already fakefied. Implies free fake tensors
        are allowed on `make_fx`.

        **kwargs: Arbitrary keyword arguments to be passed to the function f.

    Returns:
        A tuple of (graph, guards)
        Graph: An FX graph representing the execution of the input PyTorch function with the provided arguments and options.
        Guards: The guards we accumulated during tracing f above

    Raises:
        AssertionError: If decomposition_table is specified without setting aten_graph=True,
        or if graph breaks during tracing in export.

        AssertionError: If Dynamo input and output is not consistent with traced input/output.

    Note - this headerdoc was authored by ChatGPT, with slight modifications by the author.
    """
    check_if_dynamo_supported()
    torch._C._log_api_usage_once("torch._dynamo.export")
    if decomposition_table is not None:
        assert (
            aten_graph
        ), "Specifying a decomposition_table table or tracing mode is illegal without setting aten_graph=True"
    if pre_dispatch:
        assert aten_graph, "pre_dispatch=True can only be used when aten_graph=True"
    f = innermost_fn(f)
    call_to_inspect = f.forward if isinstance(f, torch.nn.Module) else f
    original_signature = inspect.signature(call_to_inspect)

    graph = None
    out_guards = None
    graph_captured_input = None
    graph_captured_result: Optional[Tuple[torch.Tensor, ...]] = None
    _allow_fake_constant: bool = (
        fake_mode is not None
    )  # Allow fake constants during symbolic tracing

    def produce_matching(source_args, candidate_args):
        matched_elements_positions = []
        dict_of_source_args = dict()
        for i in range(0, len(source_args)):
            element_id = id(source_args[i])
            dict_of_source_args[element_id] = i

        for i in range(0, len(candidate_args)):
            arg = candidate_args[i]
            # 1-element tensor arg can be unspec int/float
            if isinstance(arg, torch.Tensor) and torch.numel(arg) == 1:
                if id(arg) in dict_of_source_args:
                    matched_elements_positions.append(dict_of_source_args[id(arg)])
                elif id(arg.item()) in dict_of_source_args:
                    matched_elements_positions.append(
                        dict_of_source_args[id(arg.item())]
                    )
                else:
                    raise AssertionError(
                        "Dynamo input/output is not consistent with traced input/output"
                    )
            else:
                assert (
                    id(arg) in dict_of_source_args
                ), "Dynamo input and output is a strict subset of traced input/output"
                matched_elements_positions.append(dict_of_source_args[id(arg)])

        return matched_elements_positions

    def guard_export_print(guards: Set[_guards.Guard]):
        nonlocal out_guards
        assert out_guards is None, "whole graph export entails exactly one guard export"
        out_guards = guards

    example_inputs = []

    def dynamo_normalization_capturing_compiler(
        gm: torch.fx.GraphModule, inner_example_inputs
    ):
        nonlocal graph
        assert (
            graph is None
        ), "Tried to emit a second graph during export. Tracing through 'f' must produce a single graph."
        graph = gm

        nonlocal fake_mode, example_inputs
        fake_mode = fake_mode or _guards.detect_fake_mode(inner_example_inputs)
        example_inputs = inner_example_inputs

        def result_capturing_wrapper(*graph_inputs):
            nonlocal graph_captured_result
            nonlocal graph_captured_input

            graph_captured_input = graph_inputs
            assert graph is not None
            graph_captured_result = graph(*graph_inputs)
            return graph_captured_result

        return result_capturing_wrapper

    flat_args, in_spec = pytree.tree_flatten((args, kwargs))

    remove_from_cache(f)
    constraint_violation_error = None
    if tracing_mode != "symbolic":
        assume_static_by_default = True
    with patch("torch._dynamo.eval_frame.most_recent_backend", None), config.patch(
        summarize_dim_constraints=True,
        specialize_int=True,
        assume_static_by_default=assume_static_by_default,
        automatic_dynamic_shapes=False,
    ), torch._guards.export_fake_mode(fake_mode):
        opt_f = optimize_assert(
            dynamo_normalization_capturing_compiler,
            hooks=Hooks(
                guard_export_fn=guard_export_print,
                guard_fail_fn=None,
            ),
            export=True,
            export_constraints=constraints,
        )(f)
        # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideffects and reject.
        try:
            result_traced = opt_f(*args, **kwargs)
        except ConstraintViolationError as e:
            constraint_violation_error = e
    remove_from_cache(f)

    if (
        (shape_env := getattr(fake_mode, "shape_env", None)) is not None
        and (dim_constraints := shape_env.dim_constraints) is not None
        and not skipfiles.check(inspect.getsourcefile(call_to_inspect))
    ):
        dim_constraints.solve()
        msg = dim_constraints.prettify_results(original_signature)
        forced_specializations = dim_constraints.forced_specializations()
        if forced_specializations:
            msg = (
                "Some dynamic dimensions need to be specialized because "
                "the constraints inferred for them are too complex to specify.\n"
                f"{forced_specializations}\n{msg}"
            )
        if constraint_violation_error:
            constraint_violation_error.args = (
                constraint_violation_error.args[0] + msg,
            )
        else:
            if forced_specializations:
                constraint_violation_error = ConstraintViolationError(msg)
            else:
                log.info(
                    "Summary of dimension constraints:%s",
                    msg,
                )

        # Error if we have any constraints on static values
        for k in shape_env.var_to_range.keys():
            if isinstance(k, sympy.Integer):
                constraint_violation_error = ConstraintViolationError(
                    f"{''.join(traceback.format_list(shape_env.var_to_stack[k]))}\n"
                    "It appears that you're trying to set a constraint on a "
                    f"value which we evaluated to have a static value of {k}. "
                    "Scroll up to see where this constraint was set."
                )
    if constraint_violation_error:
        raise constraint_violation_error

    assert (
        graph is not None
    ), "Failed to produce a graph during tracing. Tracing through 'f' must produce a single graph."
    assert out_guards is not None, "Failed to produce guards during tracing"
    assert fake_mode is not None

    matched_input_elements_positions = produce_matching(flat_args, graph_captured_input)

    # NB: This is mostly hitting the cache; Dynamo already converted these
    example_fake_inputs = [fake_mode.from_tensor(t) for t in example_inputs]
    flat_results_traced, out_spec_traced = pytree.tree_flatten(result_traced)

    assert graph_captured_result is not None
    flat_both = list(graph_captured_result) + flat_args
    matched_output_elements_positions = produce_matching(flat_both, flat_results_traced)

    if aten_graph:
        # Running graph with interpreter is needed for propagating the stack_trace
        def graph_with_interpreter(*args):
            with torch.fx.traceback.preserve_node_meta():
                return torch.fx.Interpreter(graph).run(*args)

        with enable_python_dispatcher(), fake_mode:
            try:
                graph = make_fx(
                    graph_with_interpreter,
                    decomposition_table=decomposition_table,
                    tracing_mode="real",
                    _allow_non_fake_inputs=True,
                    pre_dispatch=pre_dispatch,
                    _allow_fake_constant=_allow_fake_constant,
                )(*example_fake_inputs)
            except CondOpArgsMismatchError as e:
                # Wrap the internal error to the user-facing error
                raise UserError(UserErrorType.DYNAMIC_CONTROL_FLOW, str(e))

    new_graph = FlattenInputOutputSignature(
        graph,
        flat_args,
        matched_input_elements_positions,
        matched_output_elements_positions,
        example_fake_inputs,
        fake_mode,
    ).transform()

    # Store constraints and inputs as metadata for user passes, e.g. turn constraints to runtime check
    new_graph.meta["input_shape_constraints"] = (
        [constraint.serializable_spec for constraint in constraints]
        if constraints
        else []
    )

    def signature_to_fullargspec(sig: inspect.Signature):
        # Get a list of Parameter objects from the Signature object
        params = list(sig.parameters.values())
        # Separate positional arguments, keyword-only arguments and varargs/varkw
        args = [
            p.name for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        kwonlyargs = [
            p.name for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        varargs = next(
            (p.name for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL), None
        )
        varkw = next(
            (p.name for p in params if p.kind == inspect.Parameter.VAR_KEYWORD), None
        )
        # Get default values for positional arguments and keyword-only arguments
        defaults = tuple(
            p.default
            for p in params
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and p.default is not inspect.Parameter.empty
        )
        kwonlydefaults = {
            p.name: p.default
            for p in params
            if p.kind == inspect.Parameter.KEYWORD_ONLY
            and p.default is not inspect.Parameter.empty
        }
        # Get annotations for parameters and return value
        annotations = {}
        if sig.return_annotation:
            annotations = {"return": sig.return_annotation}
        for parameter in params:
            annotations[parameter.name] = parameter.annotation
        # Return a FullArgSpec object with the extracted attributes
        return inspect.FullArgSpec(
            args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations
        )

    # Make dynamo graph to have same input/output spec as user code
    def argument_names(f: Callable[..., Any], *args, **kwargs) -> List[str]:
        fullargspec = signature_to_fullargspec(original_signature)

        # 1. Map `args` 1-to-1 to positional arguments in original signature.
        input_strs = fullargspec.args[: len(args)]

        if len(args) > len(fullargspec.args):
            # 2. If there are more arguments left in `args`, they map to varargs in original
            # signature. Assign names as {varargs}_0, {varargs}_1, ...
            assert fullargspec.varargs is not None, "More arguments than expected"
            input_strs += [
                f"{fullargspec.varargs}_{i}"
                for i in range(0, len(args) - len(input_strs))
            ]
        elif len(args) < len(fullargspec.args):
            # 3. If there are fewer arguments in `args` than `fullargspec.args`,
            # it implies these are arguments either with default values, or provided in
            # `kwargs`. The former can be safely ignored. Because Dynamo.export does not
            # export them as part of the function signature. The latter will be handled
            # in the next step.
            for unprovided_arg in fullargspec.args[
                len(args) : -len(fullargspec.defaults or [])
            ]:
                assert unprovided_arg in kwargs, f"Missing argument {unprovided_arg}"

        # 4. Keyword arguments provided in `kwargs`.
        input_strs += list(kwargs.keys())

        # 5. Keyword-only arguments with default values if not provided are not exported
        # as part of the function signature.
        for kwonly_arg in fullargspec.kwonlyargs:
            kwonlydefaults = fullargspec.kwonlydefaults or {}
            assert (
                kwonly_arg in kwargs or kwonly_arg in kwonlydefaults
            ), f"Missing keyword only argument {kwonly_arg}"

        return input_strs

    new_graph.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            argument_names(f, *args, **kwargs),
            in_spec,
            out_spec_traced,
        )
    )

    new_graph.recompile()
    return (new_graph, out_guards)
