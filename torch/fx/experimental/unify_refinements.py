from unification import unify, Var
from torch.fx.tensor_type import TensorType
from copy import deepcopy


def convert_eq(list_of_eq):
    lhs = []
    rhs = []
    for eq in list_of_eq:
        lhs.append(eq.lhs)
        rhs.append(eq.rhs)
    return tuple(lhs), tuple(rhs)


def unify_eq(list_of_eq):
    lhs, rhs = convert_eq(list_of_eq)
    return unify(lhs, rhs)


def substitute_solution_one_type(mapping, t):
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
    old_graph = deepcopy(graph)
    while True:
        for n in graph.nodes:
            n.type = substitute_solution_one_type(mapping, n.type)
        if check_for_type_equality(old_graph, graph):
            break
        else:
            old_graph = deepcopy(graph)


def check_for_type_equality(g1, g2):
    for n,m in zip(g1.nodes, g2.nodes):
        if n.type != m.type:
            return False
    return True

