import inspect
from collections import defaultdict
from typing import Dict, Tuple

import torch
from torch._dynamo.source import (
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
from torch.export.custom_obj import ScriptObjectMeta
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    DimDynamic,
    EqualityConstraint,
    ShapeEnv,
    StatelessSymbolicContext,
)


def fakify(mode, t, t_constraints, source, sources):
    """
    Make a fake tensor, given a fake mode, a tensor, constraints indexed
    by tensor ids, the source for the tensor, and an accumulator mapping
    tensor dimensions to their sources.
    """
    if t is None or isinstance(t, torch.ScriptObject):
        return t
    if not isinstance(t, torch.Tensor):
        raise ValueError("Only tensors allowed as input")

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
            sources[(t_id, i)] = src
            mode.shape_env.source_name_to_debug_name[src.name()] = constraint.debug_name
    fake = mode.from_tensor(t, source=source, symbolic_context=symbolic_context)
    mode.shape_env.tracked_fakes.append(TrackedFake(fake, source, symbolic_context))
    return fake


def fake_tree(mode, arg, t_constraints, source, sources):
    """
    Call fakify while recursively mapping on lists and dictionaries. Using pytree map
    would be ideal here, but we are also building sources as we recurse.
    """
    if isinstance(arg, (tuple, list)):
        return [
            fake_tree(mode, arg, t_constraints, GetItemSource(source, i), sources)
            for i, arg in enumerate(arg)
        ]
    elif isinstance(arg, dict):
        return {
            k: fake_tree(mode, arg, t_constraints, GetItemSource(source, k), sources)
            for k, arg in arg.items()
        }
    # TODO(avik): data classes
    else:
        return fakify(mode, arg, t_constraints, source, sources)


def make_fake_inputs(nn_module, args, constraints):
    """
    Given an nn module, example inputs, and constraints, return a new fake mode,
    fake inputs created in that mode whose dynamic shape dimensions are constrained
    by the given ranges, and sources for pairs of dynamic shape dimensions that are
    constrained to be equal.
    """
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
    with FakeTensorMode(
        shape_env=ShapeEnv(tracked_fakes=[], co_fields=co_fields)
    ) as fake_mode:
        original_signature = inspect.signature(nn_module.forward)
        params = original_signature.parameters
        sources: Dict[Tuple[int, int], Source] = {}
        fake_args = tuple(
            fake_tree(fake_mode, arg, t_constraints, LocalSource(x), sources)
            for x, arg in zip(params, args)
        )
        src_equalities = []
        for constraint in constraints:
            if constraint.shared is not None:
                src_equality = (
                    sources[(constraint.t_id, constraint.dim)],
                    sources[(constraint.shared.t_id, constraint.shared.dim)],
                )
                src_equalities.append(src_equality)
        return fake_mode, fake_args, src_equalities, original_signature


def make_constraints(fake_mode, src_equalities, original_signature, gm):
    """
    Given a fake mode, sources pairs corresponding to equal dynamic shape dimensions,
    and a graph module, produce guards on the fake mode's shape env (raising constraint
    violations if any), solve (to suggest simplifications or fixes), and return the
    resulting range constraints and equality constraints.
    """
    shape_env = fake_mode.shape_env
    placeholders = [tf.fake for tf in shape_env.tracked_fakes]
    sources = [tf.source for tf in shape_env.tracked_fakes]
    input_contexts = [tf.symbolic_context for tf in shape_env.tracked_fakes]
    equalities_inputs = EqualityConstraint(source_pairs=src_equalities, warn_only=False)
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

    range_constraints = {}
    input_dims = defaultdict(list)
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if node.meta["val"] is None or isinstance(node.meta["val"], ScriptObjectMeta):
            continue
        for i, d in enumerate(node.meta["val"].shape):
            if isinstance(d, torch.SymInt):
                range_constraints[d.node.expr] = shape_env.var_to_range[d.node.expr]
                input_dims[d.node.expr].append(InputDim(input_name=node.name, dim=i))

    equality_constraints = []
    for equal_input_dims in input_dims.values():
        primary, *others = equal_input_dims
        for other in others:
            equality_constraints.append((primary, other))

    return range_constraints, equality_constraints
