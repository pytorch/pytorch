import inspect
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch._dynamo.source import (
    AttrSource,
    GetItemSource,
    LocalSource,
    TensorProperty,
    TensorPropertySource,
)
from torch._dynamo.variables.builder import TrackedFake
from torch._export.passes.add_runtime_assertions_for_constraints_pass import InputDim
from torch._guards import Source
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import Constraint
from torch.export.dynamic_shapes import _Dim
from torch.export.exported_program import InputKind
from torch.export.graph_signature import CustomObjArgument, InputSpec, TensorArgument
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    DimDynamic,
    EqualityConstraint,
    ShapeEnv,
    StatelessSymbolicContext,
    ValueRanges,
)
from torch.utils._pytree import (
    GetAttrKey,
    KeyPath,
    MappingKey,
    SequenceKey,
    tree_flatten,
    tree_map_with_path,
)


def key_path_to_source(kp: KeyPath) -> Source:
    """
    Given a key path, return the source for the key path.
    """
    source: Source = LocalSource("args")
    for k in kp:
        if isinstance(k, SequenceKey):
            source = GetItemSource(source, k.idx)
        elif isinstance(k, MappingKey):
            source = GetItemSource(source, k.key)
        elif isinstance(k, GetAttrKey):
            source = AttrSource(source, k.name)
        else:
            raise ValueError(f"Unknown KeyEntry {k}")

    return source


def _is_constant_argument(t):
    return t is None or isinstance(t, (int, float, bool, str))


def fakify(
    mode: FakeTensorMode,
    kp: KeyPath,
    t: Any,
    t_constraints: Dict[int, Dict[int, Constraint]],
    sources: Dict[Tuple[int, int], List[Source]],
):
    source = key_path_to_source(kp)
    if _is_constant_argument(t) or isinstance(t, torch.ScriptObject):
        return t
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"Unsupported input type {type(t)}")
    n_dims = len(t.shape)
    symbolic_context = StatelessSymbolicContext(
        dynamic_sizes=[DimDynamic.STATIC] * n_dims,
        constraint_sizes=[None] * n_dims,
    )
    t_id = id(t)
    if t_id in t_constraints:
        for i, constraint in t_constraints[t_id].items():
            symbolic_context.constraint_sizes[i] = constraint.constraint_range
            symbolic_context.dynamic_sizes[i] = DimDynamic.DYNAMIC
            src = TensorPropertySource(base=source, prop=TensorProperty.SIZE, idx=i)
            sources[(t_id, i)].append(src)
            mode.shape_env.source_name_to_debug_name[src.name()] = constraint.debug_name  # type: ignore[assignment]
    fake = mode.from_tensor(t, source=source, symbolic_context=symbolic_context)
    mode.shape_env.tracked_fakes.append(TrackedFake(fake, source, symbolic_context))  # type: ignore[union-attr]
    return fake


def make_fake_params_buffers(
    fake_mode: FakeTensorMode,
    params_buffers: Dict[str, torch.Tensor],
) -> Dict[str, Union[torch.Tensor, torch.nn.Parameter]]:
    faked_params_buffers = {}
    for key, value in params_buffers.items():
        faked_params_buffers[key] = fake_mode.from_tensor(value, static_shapes=True)
    return faked_params_buffers


def make_fake_inputs(nn_module, args, kwargs, dynamic_shapes):
    """
    Given an nn module, example inputs, and constraints, return a new fake mode,
    fake inputs created in that mode whose dynamic shape dimensions are constrained
    by the given ranges, and sources for pairs of dynamic shape dimensions that are
    constrained to be equal.
    """
    # TODO(avik): refactor Dynamo to avoid duplication of the following code
    # between non-strict and strict.
    # Specifically, here (non-strict) we do the following pre-tracing steps:
    #   - Fakify inputs.
    #   - Process input shape equalities.
    # In strict, these steps are spread across multiple files:
    #   - output_graph.py fakifies inputs.
    #   - [post-tracing] guards.py processes input shape equalities.

    constraints = torch.export.dynamic_shapes._process_dynamic_shapes(
        nn_module, args, kwargs, dynamic_shapes
    )
    constraints = constraints or []
    t_constraints: Dict[int, Dict[int, Constraint]] = defaultdict(dict)
    for constraint in constraints:
        t_constraints[constraint.t_id][constraint.dim] = constraint
        if constraint.shared is not None:
            t_constraints[constraint.shared.t_id][constraint.shared.dim] = constraint

    code = nn_module.forward.__code__
    co_fields = {
        "co_name": code.co_name,
        "co_filename": code.co_filename,
        "co_firstlineno": code.co_firstlineno,
    }

    fake_mode = FakeTensorMode(
        shape_env=ShapeEnv(tracked_fakes=[], co_fields=co_fields),
        allow_non_fake_inputs=True,
    )
    if fake_mode.shape_env is None or fake_mode.shape_env.tracked_fakes is None:
        raise ValueError(
            "Detected fake_mode does not have a shape_env with tracked fakes. "
            "If you constructed the module under a FakeTensorMode, "
            "please initialize it like: FakeTensorMode(shape_env=ShapeEnv(tracked_fakes=[]))"
        )

    with fake_mode:
        original_signature = inspect.signature(nn_module.forward)
        sources: Dict[Tuple[int, int], List[Source]] = defaultdict(list)
        fake_args, fake_kwargs = tree_map_with_path(
            lambda kp, val: fakify(fake_mode, kp, val, t_constraints, sources),
            (args, kwargs),
        )

        from sympy import Symbol

        source_pairs: List[Tuple[Source, Source]] = []
        derived_equalities: List[Tuple[Source, Union[Source, Symbol], Callable]] = []
        phantom_symbols: Dict[str, Symbol] = {}
        for constraint in constraints:
            torch.export.dynamic_shapes._process_equalities(
                constraint,
                lambda t_id, dim: sources[(t_id, dim)],
                fake_mode.shape_env,
                source_pairs,
                derived_equalities,
                phantom_symbols,
            )

        equalities_inputs = EqualityConstraint(
            source_pairs=source_pairs,
            derived_equalities=derived_equalities,
            phantom_symbols=list(phantom_symbols.values()),
            warn_only=False,
        )
        return fake_mode, fake_args, fake_kwargs, equalities_inputs, original_signature


def make_constraints(
    fake_mode: FakeTensorMode,
    equalities_inputs: EqualityConstraint,
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any]],
    input_specs: List[InputSpec],
    original_signature: inspect.Signature,
    gm: torch.fx.GraphModule,
):
    """
    Given a fake mode, sources pairs corresponding to equal dynamic shape dimensions,
    and a graph module, produce guards on the fake mode's shape env (raising constraint
    violations if any), solve (to suggest simplifications or fixes), and return the
    resulting range constraints and equality constraints.
    """
    # TODO(avik): refactor Dynamo to avoid duplication of the following code
    # between non-strict and strict.
    # Specifically, here (non-strict) we do the following post-tracing steps:
    #   - Produce guards.
    #   - Solve constraints.
    #   - Install shape metadata in IR.
    # In strict, these steps are spread across multiple files:
    #   - guards.py produces guards.
    #   - eval_frame.py solves constraints
    #   - _trace.py installs shape metadata in IR.

    inline_constraints = gm.meta.get("inline_constraints", [])
    range_constraints = {
        symbol: inline_constraints[symbol] for symbol in inline_constraints
    }
    if dynamic_shapes == []:
        return range_constraints

    def _is_dynamic_shape_leaf(x):
        if x is None:
            return True
        if isinstance(x, dict):
            x = list(x.values())
        return all(isinstance(y, (_Dim, int)) or y is None for y in x)

    flat_dynamic_shapes, _ = tree_flatten(
        dynamic_shapes, is_leaf=_is_dynamic_shape_leaf
    )

    shape_env = fake_mode.shape_env
    assert shape_env.tracked_fakes is not None
    placeholders = [tf.fake for tf in shape_env.tracked_fakes]
    sources = [tf.source for tf in shape_env.tracked_fakes]
    input_contexts = [tf.symbolic_context for tf in shape_env.tracked_fakes]
    constraint_violation_error = None
    try:
        shape_env.produce_guards(
            placeholders,
            sources,
            input_contexts=input_contexts,
            equalities_inputs=equalities_inputs,
            ignore_static=False,
        )
    except ConstraintViolationError as e:
        constraint_violation_error = e

    shape_env.frozen = True
    dim_constraints = shape_env.dim_constraints
    if dim_constraints is None:
        # Expected when shape_env.produce_guards throws an early constraint violation error.
        # There is nothing to solve for in this case.
        # TODO(avik): Maybe record the constraint violation error instead and replay later?
        assert constraint_violation_error
        raise constraint_violation_error
    dim_constraints.solve()
    dim_constraints.remove_redundant_dynamic_results()
    forced_specializations = dim_constraints.forced_specializations()
    msg = dim_constraints.prettify_results(
        original_signature, constraint_violation_error, forced_specializations
    )
    if constraint_violation_error:
        constraint_violation_error.args = (constraint_violation_error.args[0] + msg,)
    elif forced_specializations:
        constraint_violation_error = ConstraintViolationError(msg)
    if constraint_violation_error:
        raise constraint_violation_error

    user_tensor_input_names = {
        spec.arg.name
        for spec in input_specs
        if spec.kind == InputKind.USER_INPUT and isinstance(spec.arg, TensorArgument)
    }

    input_dims = defaultdict(list)
    free_symbols = set()
    input_index = 0
    for node in gm.graph.nodes:
        if node.name not in user_tensor_input_names:
            continue
        if _is_constant_argument(node.meta["val"]) or isinstance(
            node.meta["val"], CustomObjArgument
        ):
            continue
        shape_spec = flat_dynamic_shapes[input_index]
        for i, d in enumerate(node.meta["val"].shape):
            if isinstance(d, torch.SymInt):
                # Look up the range constraint for the symbol corresponding to this shape dimension
                # and store it indexed by the symbolic expression corresponding to it.
                # NOTE(avik): Use node._expr instead of node.expr for the lookup here because
                # we want the symbol, not its replacement, which could be an expression. Maybe
                # there's a better way to do this, e.g., by (re)computing value ranges for expressions?
                dim = shape_spec[i] if shape_spec else None
                if dim:
                    range_constraints[d.node.expr] = ValueRanges(
                        lower=dim.min, upper=dim.max
                    )
                else:
                    range_constraints[d.node.expr] = shape_env.var_to_range[
                        d.node._expr
                    ]
                input_dims[d.node.expr].append(InputDim(input_name=node.name, dim=i))
                free_symbols.update(d.node.expr.free_symbols)
        input_index += 1

    for symbol in free_symbols:
        if symbol not in range_constraints:
            # Placeholders can have symbolic shapes that are derived expressions.
            # The above code will record direct range constraints for them
            # so that we can do runtime assertions. In addition, for serde checks
            # we want to record range constraints for their root symbols.
            range_constraints[symbol] = shape_env.var_to_range[symbol]

    return range_constraints
