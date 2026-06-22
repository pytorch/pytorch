from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ApplyBroadcasting",
    "BinConstraintD",
    "BinConstraintT",
    "BinaryConstraint",
    "BVar",
    "CalcConv",
    "CalcMaxPool",
    "CalcProduct",
    "CanReshape",
    "Conj",
    "Constraint",
    "DGreatestUpperBound",
    "Disj",
    "DVar",
    "F",
    "GetItem",
    "GetItemTensor",
    "IndexSelect",
    "Prod",
    "T",
    "TGreatestUpperBound",
    "Transpose",
    "TVar",
    "is_algebraic_expression",
    "is_bool_expr",
    "is_dim",
]

from torch.fx.experimental.migrate_gradual_types.operation import (
    op_add,
    op_div,
    op_eq,
    op_gt,
    op_lt,
    op_mod,
    op_mul,
    op_neq,
    op_sub,
)
from torch.fx.tensor_type import _DynType, Dyn, TensorType


class Constraint:
    pass


class Conj(Constraint):
    def __init__(self, conjuncts: Sequence[Constraint]) -> None:
        """
        :param conjuncts: Conjunction of constraints
        """
        self.conjucts = list(conjuncts)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Conj):
            return self.conjucts == other.conjucts
        else:
            return False

    def __repr__(self) -> str:
        return f"And({self.conjucts})"


class Disj(Constraint):
    def __init__(self, disjuncts: Sequence[Constraint]) -> None:
        """
        :param disjuncts: Disjunction of constraints
        """
        self.disjuncts = list(disjuncts)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Disj):
            return self.disjuncts == other.disjuncts
        else:
            return False

    def __repr__(self) -> str:
        return f"Or({self.disjuncts})"


class Prod(Constraint):
    def __init__(self, products: Sequence[DVar | int | _DynType]) -> None:
        """
        :param products: lists of dimensions to multiply
        """
        self.products = list(products)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Prod):
            return self.products == other.products
        else:
            return False

    def __repr__(self) -> str:
        return f"Product({self.products})"


class T(Constraint):
    """
    True
    """

    def __init__(self) -> None:
        pass

    def __eq__(self, other: object) -> bool:
        return isinstance(other, T)

    def __repr__(self) -> str:
        return "True"


class F(Constraint):
    """
    False
    """

    def __init__(self) -> None:
        pass

    def __eq__(self, other: object) -> bool:
        return isinstance(other, F)

    def __repr__(self) -> str:
        return "False"


class BinaryConstraint(Constraint):
    """
    Represents all binary operations
    """

    def __init__(self, lhs: _Operand, rhs: _Operand, op: str | None) -> None:
        """
        :param lhs: lhs of the constraint
        :param rhs: rhs of the constraint
        :param op: string representing the operation
        """
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BinaryConstraint):
            return (
                self.lhs == other.lhs and self.rhs == other.rhs and self.op == other.op
            )
        else:
            return False

    def __repr__(self) -> str:
        return f"({self.lhs} {self.op} {self.rhs})"


class BinConstraintT(BinaryConstraint):
    """
    Binary constraints about tensors
    """

    def __init__(self, lhs: _Operand, rhs: _Operand, op: str | None) -> None:
        if not (
            (isinstance(lhs, (TVar, TensorType, int)) or lhs == Dyn)
            and (isinstance(rhs, (TVar, TensorType, int)) or rhs == Dyn)
        ):
            raise AssertionError(f"Invalid types: lhs={type(lhs)}, rhs={type(rhs)}")
        super().__init__(lhs, rhs, op)


class BinConstraintD(BinaryConstraint):
    """
    Binary constraints about dimensions
    """

    def __init__(self, lhs: _Operand, rhs: _Operand, op: str | None) -> None:
        if not (is_algebraic_expression(lhs) or is_dim(lhs) or is_bool_expr(lhs)):
            raise AssertionError(f"Invalid lhs type: {type(lhs)}")
        if not (is_algebraic_expression(rhs) or is_dim(rhs) or is_bool_expr(rhs)):
            raise AssertionError(f"Invalid rhs type: {type(rhs)}")

        super().__init__(lhs, rhs, op)


class TGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for tensors with dynamic type
    """

    def __init__(self, res: TVar, rhs1: TVar, rhs2: TVar) -> None:
        """
        :param res: tensor variable that stores the result of the output
        :param rhs1: tensor or tensor variable
        :param rhs2: tensor or tensor variabke
        """
        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self) -> str:
        return f"{self.res} = {self.rhs1}\u2294*{self.rhs2}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TGreatestUpperBound):
            return (
                self.res == other.res
                and self.rhs1 == other.rhs1
                and self.rhs2 == other.rhs2
            )
        else:
            return False


class DGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for dimensions
    """

    def __init__(
        self,
        res: DVar | int | _DynType,
        rhs1: DVar | int | _DynType,
        rhs2: DVar | int | _DynType,
    ) -> None:
        """
        :param res: Dimension variable to store the result
        :param rhs1: dimension variable 1
        :param rhs2: dimension variable 2
        """
        if not is_dim(res):
            raise AssertionError(f"Expected dimension for res, got {type(res)}")
        if not is_dim(rhs1):
            raise AssertionError(f"Expected dimension for rhs1, got {type(rhs1)}")
        if not is_dim(rhs2):
            raise AssertionError(f"Expected dimension for rhs2, got {type(rhs2)}")

        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self) -> str:
        return f"{self.res} = {self.rhs1}\u2294{self.rhs2}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DGreatestUpperBound):
            return (
                self.res == other.res
                and self.rhs1 == other.rhs1
                and self.rhs2 == other.rhs2
            )
        else:
            return False


class CanReshape(Constraint):
    """
    can_reshape constraint
    """

    def __init__(self, src: TVar, target: TensorType) -> None:
        """
        :param src: tensor variable
        :param target: tensor
        """
        self.src = src
        self.target = target

    def __repr__(self) -> str:
        return f"can-reshape({self.src}, {self.target})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CanReshape):
            return self.src == other.src and self.target == other.target
        else:
            return False


class IndexSelect(Constraint):
    def __init__(
        self,
        tensor_size: int,
        input_var: TVar,
        dim_replace: DVar | _DynType,
        index: int,
        output: TVar,
    ) -> None:
        """
        Args:
            input_var: input to index_select
            tensor_size: tensor size we are considering
            dim_replace: the dimension of the output at "index"
            index: location of the dimensions to replace in the input
            output: variable to store the result
        """
        if not isinstance(input_var, TVar):
            raise AssertionError(f"Expected TVar, got {type(input_var)}")
        if not isinstance(output, TVar):
            raise AssertionError(f"Expected TVar, got {type(output)}")
        if not (isinstance(dim_replace, DVar) or dim_replace == Dyn):
            raise AssertionError(f"Expected DVar or Dyn, got {type(dim_replace)}")
        if not isinstance(index, int):
            raise AssertionError(f"Expected int, got {type(index)}")

        self.input_var = input_var
        self.tensor_size = tensor_size
        self.dim_replace = dim_replace
        self.index = index
        self.output = output

    def __repr__(self) -> str:
        return (
            f" {self.output} = "
            f"IndexSelect({self.input_var}, "
            f"tensor_size: {self.tensor_size}, "
            f"{self.dim_replace}, "
            f"{self.index})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IndexSelect):
            return (
                self.tensor_size == other.tensor_size
                and self.dim_replace == other.dim_replace
                and self.index == other.index
                and self.output == other.output
                and self.input_var == other.input_var
            )
        else:
            return False


class Transpose(Constraint):
    def __init__(
        self, tensor_size: int, input_var: TVar, index1: int, index2: int, output: TVar
    ) -> None:
        """
        Args:
            tensor_size: current tensor size
            input_var: variable to hold input
            index1: dimension 1
            index2: dimension 2
            output: output that stores result
        """
        if not isinstance(input_var, TVar):
            raise AssertionError(f"Expected TVar, got {type(input_var)}")
        if not isinstance(output, TVar):
            raise AssertionError(f"Expected TVar, got {type(output)}")
        if not isinstance(index1, int):
            raise AssertionError(f"Expected int, got {type(index1)}")
        if not isinstance(index2, int):
            raise AssertionError(f"Expected int, got {type(index2)}")

        self.input_var = input_var
        self.tensor_size = tensor_size
        self.index1 = index1
        self.index2 = index2
        self.output = output

    def __repr__(self) -> str:
        return (
            f" {self.output} = "
            f"Transpose({self.input_var}, "
            f"tensor_size: {self.tensor_size}, "
            f"{self.index1}, "
            f"{self.index2})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Transpose):
            return (
                self.tensor_size == other.tensor_size
                and self.index1 == other.index1
                and self.index2 == other.index2
                and self.output == other.output
                and self.input_var == other.input_var
            )
        else:
            return False


class GetItem(Constraint):
    def __init__(
        self, tensor_size: int, index: int, res: DVar, input_var: TVar
    ) -> None:
        """
        Constraint for getting item given a tensor size
        :param tensor_size: actual number
        :param index: actual number representing the index
        :param res: dimension variable to carry the item we get
        :param input_var: a tensor variable from which we will get item
        """
        if not isinstance(res, DVar):
            raise AssertionError(f"Expected DVar, got {type(res)}")

        self.res = res
        self.tensor_size = tensor_size
        self.index = index
        self.input_var = input_var

    def __repr__(self) -> str:
        return f" {self.res} = GetItem({self.input_var}, tensor_size: {self.tensor_size}, {self.index})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GetItem):
            return (
                self.res == other.res
                and self.tensor_size == other.tensor_size
                and self.index == other.index
                and self.input_var == other.input_var
            )
        else:
            return False


class GetItemTensor(Constraint):
    def __init__(
        self,
        tensor_size: int,
        index_tuple: tuple[None | slice, ...],
        res: TVar,
        input_var: TVar,
    ) -> None:
        """
        Constraint for getting item given a tensor size
        However, when the argument is a tuple, we will
        expect a tensor
        :param tensor_size: actual number representing the rank
        :param index_tuple: tuple for indexing
        :param res: tensor variable to carry the item we get
        :param input_var: a tensor variable from which we will get item
        """
        if not isinstance(res, TVar):
            raise AssertionError(f"Expected TVar, got {type(res)}")

        self.res = res
        self.tensor_size = tensor_size
        self.index_tuple = index_tuple
        self.input_var = input_var

    def __repr__(self) -> str:
        return f" {self.res} = GetItemT({self.input_var}, tensor_size: {self.tensor_size}, {self.index_tuple})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GetItemTensor):
            return (
                self.res == other.res
                and self.tensor_size == other.tensor_size
                and self.index_tuple == other.index_tuple
                and self.input_var == other.input_var
            )
        else:
            return False


class CalcConv(Constraint):
    def __init__(
        self,
        conv_result: TVar,
        input_var: TVar,
        c_out: int,
        kernel: int | tuple[int, int],
        padding: int | tuple[int, int],
        stride: int | tuple[int, int],
        dilation: int | tuple[int, int],
        matching_constraint_vars: list[DVar],
    ) -> None:
        """
        :param conv_result: the convolution result
        :param input_var: input to convolution
        :param c_out: output channel type
        :param kernel: kernel tuple
        """
        self.conv_result = conv_result
        self.input_var = input_var
        self.c_out = c_out
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self) -> str:
        return (
            f"{self.conv_result} ="
            f" calc-conv({self.input_var},"
            f" {self.c_out}, {self.kernel}, "
            f"{self.padding}, {self.stride},"
            f" {self.dilation})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CalcConv):
            return (
                self.conv_result == other.conv_result
                and self.input_var == other.input_var
                and self.c_out == other.c_out
                and self.kernel == other.kernel
                and self.padding == other.padding
                and self.stride == other.stride
                and self.dilation == other.dilation
                and self.matching_constraint == other.matching_constraint
            )
        else:
            return False


class CalcMaxPool(Constraint):
    def __init__(
        self,
        maxpool_result: TVar,
        input_var: TVar,
        kernel: int | tuple[int, int],
        padding: int | tuple[int, int],
        stride: int | tuple[int, int],
        dilation: int | tuple[int, int],
        matching_constraint_vars: list[DVar],
    ) -> None:
        """
        :param maxpool_result: the result of maxpool
        :param input_var: input to convolution
        :param kernel: kernel tuple
        """
        self.maxpool_result = maxpool_result
        self.input_var = input_var
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self) -> str:
        return (
            f"{self.maxpool_result} ="
            f" calc-maxpool({self.input_var},"
            f"  {self.kernel}, "
            f"{self.padding}, {self.stride},"
            f" {self.dilation})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CalcMaxPool):
            return (
                self.maxpool_result == other.maxpool_result
                and self.input_var == other.input_var
                and self.kernel == other.kernel
                and self.padding == other.padding
                and self.stride == other.stride
                and self.dilation == other.dilation
                and self.matching_constraint == other.matching_constraint
            )
        else:
            return False


class ApplyBroadcasting(Constraint):
    def __init__(self, res1: TVar, res2: TVar, input1: TVar, input2: TVar) -> None:
        """
        :param res1: resulting tensor 1
        :param res2: resulting tensor 2
        :param input1: tensor variable 1
        :param input2: tensor variable 2
        """
        self.res1 = res1
        self.res2 = res2
        self.input1 = input1
        self.input2 = input2

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ApplyBroadcasting):
            return (
                self.res1 == other.res1
                and self.res2 == other.res2
                and self.input1 == other.input1
                and self.input2 == other.input2
            )
        else:
            return False

    def __repr__(self) -> str:
        return (
            f"{self.res1}, {self.res2} ="
            f" apply-broadcasting({self.input1},"
            f" {self.input2})"
        )


class CalcProduct(Constraint):
    """
    Given correct dimensions, calculate the product for flatten accounting for Dyn
    """

    def __init__(
        self, start: int, end: int, flattened: TVar, dims_to_flatten: list[DVar]
    ) -> None:
        """
        :param start: start index
        :param end: end index
        :param flattened: variable to store the product
        :param dims_to_flatten: the type which we will flatten
        """
        if not isinstance(dims_to_flatten, list):
            raise AssertionError(f"Expected list, got {type(dims_to_flatten)}")
        if not isinstance(flattened, TVar):
            raise AssertionError(f"Expected TVar, got {type(flattened)}")
        if not isinstance(start, int):
            raise AssertionError(f"Expected int, got {type(start)}")
        if not isinstance(end, int):
            raise AssertionError(f"Expected int, got {type(end)}")

        self.start = start
        self.end = end
        self.dims_to_flatten = dims_to_flatten
        self.flattened = flattened

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CalcProduct):
            return (
                self.start == other.start
                and self.end == other.end
                and self.dims_to_flatten == other.dims_to_flatten
                and self.flattened == other.flattened
            )

        else:
            return False

    def __repr__(self) -> str:
        return f"{self.flattened} = CalcProduct({self.start}, {self.end}, {self.dims_to_flatten})"


class TVar:
    """
    Tensor variable with no tensor constructor
    """

    def __init__(self, tvar: int) -> None:
        """
        :param tvar: tensor variable
        """
        self.tvar = tvar

    def __repr__(self) -> str:
        return f"TV({self.tvar})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TVar):
            return self.tvar == other.tvar
        else:
            return False


class DVar:
    """
    Dimension variable
    """

    def __init__(self, c: int) -> None:
        """
        :param c: character or number
        """
        self.c = c

    def __repr__(self) -> str:
        return f"DV({self.c})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DVar):
            return self.c == other.c
        else:
            return False


class BVar:
    """
    Boolean variable
    """

    def __init__(self, c: int) -> None:
        """
        :param c: character or number
        """
        self.c = c

    def __repr__(self) -> str:
        return f"BV({self.c})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BVar):
            return self.c == other.c
        else:
            return False


_Operand: TypeAlias = (
    TVar
    | TensorType
    | DVar
    | int
    | float
    | bool
    | _DynType
    | BinConstraintD
    | Prod
    | BVar
    | Conj
    | Disj
    | None
)


def is_algebraic_expression(constraint: object) -> bool:
    if isinstance(constraint, BinConstraintD):
        return constraint.op in [op_add, op_sub, op_div, op_mul, op_mod]
    else:
        return isinstance(constraint, Prod)


def is_bool_expr(constraint: object) -> bool:
    if isinstance(constraint, BinConstraintD):
        return constraint.op in [op_gt, op_lt, op_neq, op_eq]
    else:
        return isinstance(constraint, (BVar, Conj, Disj))


def is_dim(d: object) -> bool:
    return isinstance(d, (DVar, int)) or d == Dyn
