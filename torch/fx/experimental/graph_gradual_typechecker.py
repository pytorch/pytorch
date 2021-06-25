from functools import reduce
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise

# here we will collect types for a nodelist
expr_env = {}


def type_check(graph):
    """
    A gradual type checker for graphs
    Effect: every node's field type will be
    populated with a type after type-checking is done
    """
    # here we collect types for variables
    env = {}

    # populate the type environment with parameter types
    collect_params(graph.nodes, env)
    # type check every node with gradual type rules
    # if any node does not type check return false
    for n in graph.nodes:
        if not type_check_node(n, env):
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


def type_check_node(n, env):
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

            t1 = env[n.args[0].name]
            t2 = env[n.args[1].name]

            if is_consistent(t1, t2):
                expr_env[n.name] = t1
                if is_more_precise(t1, t2):
                    n.type = t1
                else:
                    n.type = t2
                expr_env[n.name] = n.type
                return n.type
            else:
                return False

        if n.name == 'reshape':
            t1 = env[n.args[0].name]
            t2 = n.args[1]
            t2_type = TensorType([Dyn if elem == -1 else elem for elem in t2])

            # if we do not know the original tensor dimension,
            # we return the required dimension
            if t1 == Dyn:
                expr_env[n.name] = t2_type
                n.type = t2_type
                return t2_type

            # if any of the dimensions are unknown,
            # we check for divisibility
            if isinstance(t1, TensorType) and Dyn in t1.__args__ or -1 in t2:
                a = [e if e != Dyn else 1 for e in t1.__args__]
                p1 = reduce(lambda x, y: x * y, a)
                p2 = reduce(lambda x, y: x * y, t2)
                if p1 % p2 == 0 or p2 % p1 == 0:
                    expr_env[n.name] = t2_type
                    n.type = t2_type
                    return t2_type
                else:
                    return False

            # if all dimensions are known we check the products
            if isinstance(t1, TensorType):
                p1 = reduce(lambda x, y: x * y, t1.__args__)
                p2 = reduce(lambda x, y: x * y, t2)
                if p1 == p2:
                    expr_env[n.name] = t2_type
                    n.type = t2_type
                    return t2_type
                else:
                    return False

            else:
                return False

        if n.name == 'transpose':
            t = env[n.args[0].name]
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
