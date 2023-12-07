import inspect
import sys
from collections import defaultdict

import sympy
import torch
import torch.utils._pytree as pytree
from torch._dynamo.source import (
    GetItemSource,
    LocalSource,
    TensorProperty,
    TensorPropertySource,
)
from torch._dynamo.variables.builder import TrackedFake
from torch._export.passes.add_runtime_assertions_for_constraints_pass import InputDim
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    EqualityConstraint,
    ShapeEnv,
    StatelessSymbolicContext,
    StrictMinMaxConstraint,
)
from torch.utils._sympy.value_ranges import ValueRanges


def fakify(mode, t, tensor_constraints, source, dim_name_to_sources):
    n_dims = len(t.shape)
    t_id = id(t)
    if t_id not in tensor_constraints:
        symbolic_context = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.STATIC] * n_dims,
            constraint_sizes=None,
        )
    else:
        symbolic_context = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.STATIC] * n_dims,
            constraint_sizes=[None] * n_dims,
        )
        for i, constraint in tensor_constraints[t_id].items():
            symbolic_context.constraint_sizes[i] = constraint.constraint_range
            symbolic_context.dynamic_sizes[i] = DimDynamic.DYNAMIC
            src = TensorPropertySource(base=source, prop=TensorProperty.SIZE, idx=i)
            dim_name_to_sources[constraint.debug_name].append(src)
            mode.shape_env.source_name_to_debug_name[src.name()] = constraint.debug_name
    fake = mode.from_tensor(t, source=source, symbolic_context=symbolic_context)
    mode.shape_env.tracked_fakes.append(
        TrackedFake(fake, source, symbolic_context.constraint_sizes)
    )
    return fake


def fake_tree(mode, arg, tensor_constraints, source, dim_name_to_sources):
    if isinstance(arg, list):
        return [
            fake_tree(
                mode,
                arg,
                tensor_constraints,
                GetItemSource(source, i),
                dim_name_to_sources,
            )
            for i, arg in enumerate(arg)
        ]

    elif isinstance(arg, dict):
        return {
            k: fake_tree(
                mode,
                arg,
                tensor_constraints,
                GetItemSource(source, k),
                dim_name_to_sources,
            )
            for k, arg in arg.items()
        }

    else:
        return fakify(mode, arg, tensor_constraints, source, dim_name_to_sources)


def make_fake_inputs(nn_module, args, constraints):
    tensor_constraints = defaultdict(dict)
    for constraint in constraints:
        tensor_constraints[constraint.t_id][constraint.dim] = constraint
        if constraint.shared is not None:
            tensor_constraints[constraint.shared.t_id][constraint.shared.dim] = constraint

    code = nn_module.forward.__code__
    co_fields = {
        "co_name": code.co_name,
        "co_filename": code.co_filename,
        "co_firstlineno": code.co_firstlineno,
    }
    with FakeTensorMode(
        shape_env=ShapeEnv(tracked_fakes=[], co_fields=co_fields)
    ) as fake_mode:
        params = inspect.signature(nn_module.forward).parameters
        dim_name_to_sources = defaultdict(list)
        fake_args = tuple(
            fake_tree(
                fake_mode, arg, tensor_constraints, LocalSource(x), dim_name_to_sources
            )
            for x, arg in zip(params, args)
        )
        return fake_mode, fake_args, dim_name_to_sources


def make_constraints(fake_mode, dim_name_to_sources, gm):
    shape_env = fake_mode.shape_env
    placeholders = [tf.fake for tf in shape_env.tracked_fakes]
    sources = [tf.source for tf in shape_env.tracked_fakes]
    constraint_inputs = [tf.constraint_dims for tf in shape_env.tracked_fakes]
    source_pairs = []
    for equal_sources in dim_name_to_sources.values():
        primary_src, *others = equal_sources
        for src in others:
            source_pairs.append((primary_src, src))
    equalities_inputs = EqualityConstraint(
        source_pairs=source_pairs, warn_only=False
    )
    shape_env.produce_guards(
        placeholders,
        sources,
        constraint_inputs=constraint_inputs,
        equalities_inputs=equalities_inputs,
        ignore_static=False,
    )
    shape_env.frozen = True
    shape_env.dim_constraints.solve()

    range_constraints = {}
    input_dims = defaultdict(list)
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        for i, d in enumerate(node.meta["val"].shape):
            if isinstance(d, torch.SymInt):
                range_constraints[d.node.expr] = shape_env.var_to_range[
                    d.node.expr
                ]
                input_dims[d.node.expr].append(
                    InputDim(input_name=node.name, dim=i)
                )

    equality_constraints = []
    for equal_input_dims in input_dims.values():
        primary, *others = equal_input_dims
        for other in others:
            equality_constraints.append((primary, other))

    return range_constraints, equality_constraints
