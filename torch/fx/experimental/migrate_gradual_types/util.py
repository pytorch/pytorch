from torch.fx.experimental.migrate_gradual_types.constraint import TVar, DVar, BinConstraintD  # type: ignore[import]
from torch.fx.experimental.migrate_gradual_types.operation import op_leq  # type: ignore[import]


def gen_tvar(curr):
    curr += 1
    return TVar(curr), curr


def gen_dvar(curr):
    curr += 1
    return DVar(curr), curr


def gen_tensor_dims(n, curr):
    dims = []
    for _ in range(n):
        dvar, curr = gen_dvar(curr)
        dims.append(dvar)
    return dims, curr


def gen_nat_constraints(list_of_dims):
    """
    Generate natural number constraints for dimensions
    """
    return [BinConstraintD(0, d, op_leq) for d in list_of_dims]
