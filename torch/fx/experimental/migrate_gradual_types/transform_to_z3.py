from torch.fx.experimental.migrate_gradual_types.constraint import Conj, Disj, T, F, BinConstraintT
from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintD, TVar, DVar
from torch.fx.experimental.migrate_gradual_types.constraint import Prod, is_algebraic_expression, is_dim
from torch.fx.experimental.migrate_gradual_types.constraint_generator import ConstraintGenerator
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import transform_constraint
from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_eq, op_neq, op_gt, op_lt
from torch.fx.experimental.migrate_gradual_types.operation import op_leq, op_sub, op_div, op_mul, op_mod
from torch.fx.experimental.migrate_gradual_types.z3_types import tensor_type, z3_dyn, D
from torch.fx.tensor_type import TensorType, Dyn

try:
    import z3  # type: ignore[import]
    HAS_Z3 = True

    def transform_to_z3(constraint, counter, dimension_dict):
        if isinstance(constraint, Conj):
            conjuncts = []
            for c in constraint.conjucts:
                new_c, counter = transform_to_z3(c, counter, dimension_dict)
                conjuncts.append(new_c)
            return z3.And(conjuncts), counter

        elif isinstance(constraint, Disj):
            disjuncts = []
            for c in constraint.disjuncts:
                new_c, counter = transform_to_z3(c, counter, dimension_dict)
                disjuncts.append(new_c)
            return z3.Or(disjuncts), counter

        elif isinstance(constraint, T):
            return True, counter

        elif isinstance(constraint, F):
            return False, counter

        elif isinstance(constraint, BinConstraintT):
            assert constraint.op == op_eq
            lhs, counter = transform_var(constraint.lhs, counter, dimension_dict)
            rhs, counter = transform_var(constraint.rhs, counter, dimension_dict)
            return (lhs == rhs), counter

        elif isinstance(constraint, BinConstraintD):

            lhs, counter = transform_algebraic_expression(constraint.lhs, counter, dimension_dict)
            rhs, counter = transform_algebraic_expression(constraint.rhs, counter, dimension_dict)

            if constraint.op == op_eq:
                if is_dim(constraint.lhs) and is_dim(constraint.rhs):
                    lhs, counter = transform_dimension(constraint.lhs, counter, dimension_dict)
                    rhs, counter = transform_dimension(constraint.rhs, counter, dimension_dict)
                    return lhs == rhs, counter

                else:
                    # otherwise, we consider algebraic expressions
                    return lhs == rhs, counter

            # The assumption here is that the LHS and RHS must be dimensions
            elif constraint.op == op_neq:
                assert is_dim(constraint.lhs)
                assert is_dim(constraint.rhs)
                lhs, counter = transform_dimension(constraint.lhs, counter, dimension_dict)
                rhs, counter = transform_dimension(constraint.rhs, counter, dimension_dict)
                if constraint.rhs == Dyn or constraint.lhs == Dyn:
                    if constraint.rhs == Dyn:
                        return lhs.arg(0) == 1, counter
                    elif constraint.lhs == Dyn:
                        return rhs.arg(0) == 1, counter

                    # return lhs.arg(0) != rhs.arg(0), counter
                # if one of the instances is a number
                elif isinstance(constraint.lhs, int) or isinstance(constraint.rhs, int):
                    if isinstance(constraint.lhs, int):
                        return z3.Or([rhs.arg(0) == 0, z3.And([rhs.arg(0) == 1, lhs.arg(1) != rhs.arg(1)])]), counter

                    elif isinstance(constraint.rhs, int):
                        return z3.Or([lhs.arg(0) == 0, z3.And([lhs.arg(0) == 1, lhs.arg(1) != rhs.arg(1)])]), counter

                    # return Or([lhs.arg(1) != rhs.arg(1), lhs.arg(0) != lhs.arg(0)]), counter
                else:
                    raise NotImplementedError

            elif constraint.op == op_leq:
                # if the dimensions are not dyn, this will come into effect
                # there would have been another constsraint specifying if a given dimension
                # is dyn or not
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                return lhs <= rhs, counter

            elif constraint.op == op_gt:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                return lhs > rhs, counter

            elif constraint.op == op_lt:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                return lhs < rhs, counter

            else:
                raise NotImplementedError('operation not yet implemented')

        else:

            # print(constraint)

            raise NotImplementedError('Operation not yet implemented')


    def transform_var(tensor, counter, dimension_dict):
        """
        Transforms tensor variables to a format understood by z3
        :param tensor: Tensor variable or a tensor type potentially with variable dimensions
        :return: Transformed variable to a z3 format
        """
        if isinstance(tensor, TensorType):
            res = []
            for t in tensor.__args__:
                transformed, counter = transform_dimension(t, counter, dimension_dict)
                res.append(transformed)

            assert len(res) <= 4
            if len(tensor.__args__) == 1:
                return tensor_type.tensor1(res[0]), counter
            elif len(tensor.__args__) == 2:
                return tensor_type.tensor2(res[0], res[1]), counter
            elif len(tensor.__args__) == 3:
                return tensor_type.tensor3(res[0], res[1], res[2]), counter
            elif len(tensor.__args__) == 4:
                return tensor_type.tensor4(res[0], res[1], res[2], res[3]), counter

        elif tensor == Dyn:
            return z3_dyn, counter

        elif isinstance(tensor, TVar):
            return z3.Const(tensor.tvar, tensor_type), counter

    def transform_dimension(dimension, counter, dimension_dict):
        """
        Takes a dimension variable or a number and transforms it to a tuple
        according to our scheme
        :param dimension: the dimension to be transformed
        :return: A tuple and the current counter
        """

        if dimension == Dyn:
            counter += 1
            return D(0, z3.Int(counter)), counter
        elif isinstance(dimension, int):
            return D(1, dimension), counter
        elif isinstance(dimension, DVar):
            if dimension.c in dimension_dict:
                return D(z3.Int(dimension_dict[dimension.c]), z3.Int(dimension.c)), counter
            else:
                raise NotImplementedError('Operation not yet implemented')


    def transform_var(tensor, counter, dimension_dict):
        """
        Transforms tensor variables to a format understood by z3
        :param tensor: Tensor variable or a tensor type potentially with variable dimensions
        :return: Transformed variable to a z3 format
        """
        if isinstance(tensor, TensorType):
            res = []
            for t in tensor.__args__:
                transformed, counter = transform_dimension(t, counter, dimension_dict)
                res.append(transformed)

            assert len(res) <= 4
            if len(tensor.__args__) == 1:
                return tensor_type.tensor1(res[0]), counter
            elif len(tensor.__args__) == 2:
                return tensor_type.tensor2(res[0], res[1]), counter
            elif len(tensor.__args__) == 3:
                return tensor_type.tensor3(res[0], res[1], res[2]), counter
            elif len(tensor.__args__) == 4:
                return tensor_type.tensor4(res[0], res[1], res[2], res[3]), counter

        elif tensor == Dyn:
            return z3_dyn, counter

        elif isinstance(tensor, TVar):
            return z3.Const(tensor.tvar, tensor_type), counter

    def transform_dimension(dimension, counter, dimension_dict):
        """
        Takes a dimension variable or a number and transforms it to a tuple
        according to our scheme
        :param dimension: the dimension to be transformed
        :return: A tuple and the current counter
        """

        if dimension == Dyn:
            counter += 1
            return D(0, z3.Int(counter)), counter
        elif isinstance(dimension, int):
            return D(1, dimension), counter
        elif isinstance(dimension, DVar):
            if dimension.c in dimension_dict:
                return D(z3.Int(dimension_dict[dimension.c]), z3.Int(dimension.c)), counter
            else:
                counter += 1
                dimension_dict[dimension.c] = counter
                return D(z3.Int(counter), z3.Int(dimension.c)), counter


    def transform_algebraic_expression(expr, counter, dimension_dict):
        """
        :param expr: An expression is either a dimension variable or an algebraic-expression
        :param counter:
        :return:
        """
        assert is_algebraic_expression(expr) or is_dim(expr)

        if is_dim(expr):
            transformed, counter = transform_dimension(expr, counter, dimension_dict)
            return transformed.arg(1), counter

        elif isinstance(expr, Prod):

            dims = []
            for dim in expr.products:
                assert is_dim(dim)
                d, counter = transform_dimension(dim, counter, dimension_dict)
                dims.append(d.arg(1))
            return z3.Product(dims), counter

        elif is_algebraic_expression(expr):

            lhs, counter = transform_algebraic_expression(expr.lhs, counter, dimension_dict)
            rhs, counter = transform_algebraic_expression(expr.rhs, counter, dimension_dict)

            if expr.op == op_sub:
                c = lhs - rhs

            elif expr.op == op_add:
                c = lhs + rhs

            elif expr.op == op_div:
                c = lhs / rhs

            elif expr.op == op_mul:
                c = lhs * rhs

            elif expr.op == op_mod:
                c = lhs % rhs

            else:
                raise NotImplementedError('opearation not yet implemented')

            return c, counter

        else:
            raise RuntimeError


    def transform_all_constraints(traced, counter=0):
        """
        :param traced: traced graph
        :param solver: solver
        :return: the satisfiability result of the constraints
        """
        dimension_dict = {}  # type: ignore[var-annotated]

        generator = ConstraintGenerator(traced)
        new_constraints, counter = generator.generate_constraints(counter)

        # print(new_constraints.conjucts[0])
        # print(*new_constraints.conjucts, sep='\n')

        # transform precision, matching, consistency till obtaining a fixed point
        old_c = None
        while old_c != new_constraints:
            old_c = new_constraints
            new_constraints, counter = transform_constraint(new_constraints, counter)

        # print(new_constraints.conjucts)
        # new_constraints.conjucts = new_constraints.conjucts[:-1]
        # print(*new_constraints.conjucts, sep='\n')

        transformed, counter = transform_to_z3(new_constraints, counter, dimension_dict)

        return transformed

except ImportError:
    HAS_Z3 = False
