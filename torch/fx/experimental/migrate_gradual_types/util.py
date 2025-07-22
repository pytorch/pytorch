from torch.fx.experimental.migrate_gradual_types.constraint import (
    BinConstraintD,
    BVar,
    DVar,
    TVar,
)
from torch.fx.experimental.migrate_gradual_types.operation import op_leq


def gen_tvar(curr: int) -> tuple[TVar, int]:
    """
    Generate a tensor variable
    :param curr: The current counter
    :return: a tensor variable and the updated counter
    """
    curr += 1
    return TVar(curr), curr


def gen_dvar(curr: int) -> tuple[DVar, int]:
    """
    Generate a dimension variable
    :param curr: the current counter
    :return: a dimension variable and an updated counter
    """
    curr += 1
    return DVar(curr), curr


def gen_bvar(curr: int) -> tuple[BVar, int]:
    """
    Generate a boolean variable
    :param curr: the current counter
    :return: a boolean variable and an updated counter
    """
    curr += 1
    return BVar(curr), curr


def gen_tensor_dims(n: int, curr: int) -> tuple[list[DVar], int]:
    """
    Generate a list of tensor dimensions
    :param n:  the number of dimensions
    :param curr: the current counter
    :return: a list of dimension variables and an updated counter
    """
    dims = []
    for _ in range(n):
        dvar, curr = gen_dvar(curr)
        dims.append(dvar)
    return dims, curr


def gen_nat_constraints(list_of_dims: list[DVar]) -> list[BinConstraintD]:
    """
    Generate natural number constraints for dimensions
    """
    return [BinConstraintD(0, d, op_leq) for d in list_of_dims]
