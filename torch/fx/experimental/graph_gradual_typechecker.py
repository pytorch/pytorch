from functools import reduce
from torch.fx.tensor_type import Dyn, is_consistent, TensorType


def type_check(graph):
    """
    A gradual type checker for graphs
    """
    env = {}
    expr_env = {}
    # populate the type environment
    collect_params(graph.nodes, env)
    for n in graph.nodes:
        if not type_check_node(n, env, expr_env):
            return False
    return True


def collect_params(node_list, env):
    """
    Collect argument types from the type environment.
    If not present, populate with Dyn
    """
    for node in node_list:
        if node.op == 'placeholder':
            if node.type:
                env[node.name] = node.type
            else:
                node.type = Dyn
                env[node.name] = node.type


def type_check_node(n, env, expr_env):
    """
    Type check a given node.
    Current operations:
    - Reshape
    - Transpose
    - Add
    """
    if n.op == 'placeholder':
        if n.name in env.keys():
            n.type = env[n.name]
            return n.type
        else:
            raise Exception("free variable")

    if n.op == 'call_function':
        if n.name == 'add':
            t1 = type_check_node(n.args[0], env, expr_env)
            t2 = type_check_node(n.args[1], env, expr_env)
            if is_consistent(t1, t2):
                expr_env[n.name] = t1
                n.type = t1
                return n.type
            else:
                return False

        if n.name == 'reshape':
            t1 = type_check_node(n.args[0], env, expr_env)
            t2 = n.args[1]
            prod1 = reduce(lambda x, y: x*y, t1.__args__)
            prod2 = reduce(lambda x, y: x*y, t2)
            if prod1 == prod2:
                expr_env[n.name] = TensorType(t2)
                n.type = TensorType(t2)
                return n.type
            else:
                return False

        if n.name == 'transpose':
            t = type_check_node(n.args[0],env, expr_env)
            dim1, dim2 = n.args[1], n.args[2]

            if 0 <= dim1 < len(t.__args__) and 0 <= dim2 < len(t.__args__):
                new_type = list(t.__args__)
                new_type[dim1], new_type[dim2] = new_type[dim2], new_type[dim1]
                final = TensorType(new_type)
                expr_env[n.name] = final
                n.type = final
                return n.type
            else:
                return False

    if n.op == 'output':
        n.type = expr_env[str(n.args[0])]
        return n.type

    else:
        return True
