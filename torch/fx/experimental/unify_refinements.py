from unification import unify, Var
from torch.fx.experimental.graph_gradual_typechecker import Refine
from torch.fx.tensor_type import TensorType
from copy import deepcopy


def infer_symbolic_types(traced):
    """
    Generate constraints over types,
    solve constraints with unification,
    apply solution back to the types
    """
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)

def convert_eq(list_of_eq):
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


def unify_eq(list_of_eq):
    """
    Apply unification to a set of
    equality constraints
    """
    lhs, rhs = convert_eq(list_of_eq)
    return unify(lhs, rhs)


def substitute_solution_one_type(mapping, t):
    """
    Apply the most general unifier to a type
    """
    if isinstance(t, Var):
        if t in mapping.keys():
            return mapping[t]

    elif isinstance(t, TensorType):
        new_type = []
        for typ in t.__args__:
            if typ in mapping.keys():
                new_type.append(mapping[typ])
            else:
                new_type.append(typ)
        return TensorType(tuple(new_type))


def substitute_all_types(graph, mapping):
    """
    Apply the most general unifier to all types in a graph
    till reaching a fixed point. If the input and output graph
    are the same, we converge.
    """
    old_graph = deepcopy(graph)
    while True:
        for n in graph.nodes:
            n.type = substitute_solution_one_type(mapping, n.type)
        if check_for_type_equality(old_graph, graph):
            break
        else:
            old_graph = deepcopy(graph)


def check_for_type_equality(g1, g2):
    """
    A check equality to be used in fixed points.
    We do not use graph equality but instead type
    equality.
    """
    for n, m in zip(g1.nodes, g2.nodes):
        if n.type != m.type:
            return False
    return True
