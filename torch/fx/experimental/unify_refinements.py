from typing import Any

import torch
import torch.fx
from torch.fx.experimental.graph_gradual_typechecker import Refine
from torch.fx.experimental.refinement_types import Equality
from torch.fx.experimental.unification import unify, Var  # type: ignore[attr-defined]
from torch.fx.tensor_type import TensorType


__all__ = [
    "check_for_type_equality",
    "convert_eq",
    "infer_symbolic_types",
    "infer_symbolic_types_single_pass",
    "substitute_all_types",
    "substitute_solution_one_type",
    "unify_eq",
]


def infer_symbolic_types_single_pass(traced: torch.fx.GraphModule) -> None:
    """
    Calls our symbolic inferencer once.
    """
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)


def infer_symbolic_types(traced: torch.fx.GraphModule) -> None:
    """
    Calls our symbolic inferencer twice.
    This is useful when one pass is not enough
    to infer all the information such as the case
    for broadcasting.
    """
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)

    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)

    r.symbolic_relations()


def convert_eq(list_of_eq: list[Equality]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """
    Convert equality constraints in the right format
    to be used by unification library.
    """
    lhs = []
    rhs = []
    for eq in list_of_eq:
        lhs.append(eq.lhs)
        rhs.append(eq.rhs)
    return tuple(lhs), tuple(rhs)


def unify_eq(list_of_eq: list[Equality]) -> Any:
    """
    Apply unification to a set of
    equality constraints
    """
    lhs, rhs = convert_eq(list_of_eq)
    return unify(lhs, rhs)


def substitute_solution_one_type(mapping: dict[object, object], t: object) -> Any:
    """
    Apply the most general unifier to a type
    """
    if isinstance(t, Var):
        if t in mapping:
            return mapping[t]
        else:
            return t

    elif isinstance(t, TensorType):
        new_type = []
        for typ in t.__args__:
            if typ in mapping:
                new_type.append(mapping[typ])
            else:
                new_type.append(typ)
        return TensorType(tuple(new_type))

    elif isinstance(t, list):
        new_type = []
        for typ in t:
            new_type.append(substitute_solution_one_type(mapping, typ))
        return new_type

    elif isinstance(t, tuple):
        new_type = []
        for typ in t:
            new_type.append(substitute_solution_one_type(mapping, typ))
        return tuple(new_type)

    else:
        return t


def substitute_all_types(graph: torch.fx.Graph, mapping: dict[object, object]) -> None:
    """
    Apply the most general unifier to all types in a graph
    till reaching a fixed point. If the input and output graph
    are the same, we converge.
    """
    flag = True
    while flag:
        flag = False
        for k in mapping:
            old_mapping_val = mapping[k]
            if mapping[k] in mapping:
                new_key = mapping[k]
                mapping[k] = mapping[new_key]
            if old_mapping_val != mapping[k]:
                flag = True

    for n in graph.nodes:
        n.type = substitute_solution_one_type(mapping, n.type)


def check_for_type_equality(g1: torch.fx.Graph, g2: torch.fx.Graph) -> bool:
    """
    A check equality to be used in fixed points.
    We do not use graph equality but instead type
    equality.
    """
    for n, m in zip(g1.nodes, g2.nodes):
        if n.type != m.type:
            return False
    return True
