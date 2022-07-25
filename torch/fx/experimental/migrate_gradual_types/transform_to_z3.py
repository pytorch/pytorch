from torch.fx.experimental.migrate_gradual_types.constraint import Conj, Disj, T, F, BinConstraintT, BVar, is_bool_expr
from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintD, TVar, DVar
from torch.fx.experimental.migrate_gradual_types.constraint import Prod, is_algebraic_expression, is_dim
from torch.fx.experimental.migrate_gradual_types.constraint_generator import ConstraintGenerator
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import transform_constraint
from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_eq, op_neq, op_gt, op_lt
from torch.fx.experimental.migrate_gradual_types.operation import op_leq, op_sub, op_div, op_mul, op_mod
from torch.fx.tensor_type import TensorType, Dyn

try:
    import z3  # type: ignore[import]
    from torch.fx.experimental.migrate_gradual_types.z3_types import tensor_type, z3_dyn, D
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
            if constraint.op == op_eq:
                lhs, counter = transform_var(constraint.lhs, counter, dimension_dict)
                rhs, counter = transform_var(constraint.rhs, counter, dimension_dict)
                return (lhs == rhs), counter

            else:
                raise NotImplementedError('Method not yet implemented')

        elif isinstance(constraint, BinConstraintD):
            if constraint.op == op_eq:

                if isinstance(constraint.lhs, BVar) and is_bool_expr(constraint.rhs):
                    transformed_rhs, counter = transform_to_z3(constraint.rhs, counter, dimension_dict)
                    transformed_lhs = z3.Bool(constraint.lhs.c)
                    return transformed_lhs == transformed_rhs, counter

                elif is_dim(constraint.lhs) and is_dim(constraint.rhs):
                    # with dimension tranformations we consider the encoding
                    lhs, counter = transform_dimension(constraint.lhs, counter, dimension_dict)
                    rhs, counter = transform_dimension(constraint.rhs, counter, dimension_dict)
                    return lhs == rhs, counter

                else:
                    # then we have an algebraic expression which means that we disregard the
                    # first element of the encoding
                    lhs, counter = transform_algebraic_expression(constraint.lhs, counter, dimension_dict)
                    rhs, counter = transform_algebraic_expression(constraint.rhs, counter, dimension_dict)
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

                # if one of the instances is a number
                elif isinstance(constraint.lhs, int) or isinstance(constraint.rhs, int):
                    if isinstance(constraint.lhs, int):
                        return z3.Or([rhs.arg(0) == 0, z3.And([rhs.arg(0) == 1, lhs.arg(1) != rhs.arg(1)])]), counter

                    elif isinstance(constraint.rhs, int):
                        return z3.Or([lhs.arg(0) == 0, z3.And([lhs.arg(0) == 1, lhs.arg(1) != rhs.arg(1)])]), counter

                else:
                    return z3.Or([z3.And([lhs.arg(0) == 0, rhs.arg(0) != 0]),
                                  z3.And([lhs.arg(0) != 0, rhs.arg(0) == 0]),
                                  z3.And([lhs.arg(0) != 0, rhs.arg(0) != 0, lhs.arg(1) != rhs.arg(1)])]), counter


            elif constraint.op == op_leq:
                # if the dimensions are not dyn, this will come into effect
                # there would have been another constraint specifying if a given dimension
                # is dyn or not
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                lhs, counter = transform_algebraic_expression(constraint.lhs, counter, dimension_dict)
                rhs, counter = transform_algebraic_expression(constraint.rhs, counter, dimension_dict)
                return lhs <= rhs, counter

            elif constraint.op == op_gt:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                lhs, counter = transform_algebraic_expression(constraint.lhs, counter, dimension_dict)
                rhs, counter = transform_algebraic_expression(constraint.rhs, counter, dimension_dict)
                return lhs > rhs, counter

            elif constraint.op == op_lt:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                lhs, counter = transform_algebraic_expression(constraint.lhs, counter, dimension_dict)
                rhs, counter = transform_algebraic_expression(constraint.rhs, counter, dimension_dict)
                return lhs < rhs, counter

            else:
                raise NotImplementedError('operation not yet implemented')

        else:
            raise NotImplementedError('Operation not yet implemented')


    def transform_var(tensor, counter, dimension_dict):
        """
        Transforms tensor variables to a format understood by z3
        Args:
            tensor: Tensor variable or a tensor type potentially with variable dimensions
        Returns: Transformed variable to a z3 format

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
        Args:
            dimension: The dimension to be transformed
            counter: variable tracking

        Returns:  tuple and the current counter

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
        Transforms an algebraic expression to z3 format
        Args:
            expr: An expression is either a dimension variable or an algebraic-expression


        Returns: the transformed expression

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
                raise NotImplementedError('operation not yet implemented')

            return c, counter

        else:
            raise RuntimeError


    def transform_all_constraints(traced, counter=0):
        """
        Given a trace, generates constraints and transforms them to z3 format

        """
        dimension_dict = {}  # type: ignore[var-annotated]

        generator = ConstraintGenerator(traced)
        new_constraints, counter = generator.generate_constraints(counter)

        # print(new_constraints.conjucts[0])
        # print(*new_constraints.conjucts, sep='\n')

        # transform precision, matching, consistency till obtaining a fixed point
        new_constraints, counter = iterate_till_fixed_point(new_constraints, counter)
        # print(new_constraints)
        # print(new_constraints.conjucts)
        # new_constraints.conjucts = new_constraints.conjucts[:-1]
        # print(*new_constraints.conjucts, sep='\n')

        transformed, counter = transform_to_z3(new_constraints, counter, dimension_dict)
        # print(transformed)
        return transformed

    def iterate_till_fixed_point(constraints, counter):
        """
        Transform constraints till reaching a fixed point
        """
        old_c = None
        while old_c != constraints:
            old_c = constraints
            constraints, counter = transform_constraint(constraints, counter)
        return constraints, counter

    def transform_all_constraints_trace_time(tracer_root, graph, node, counter=0):
        """
        Takes a node and a graph and generates two sets of constraints.
        One set constraints the node's constraints and another set
        constraints the negation of the node's constraints
        Args:
            tracer_root: the root for getting the module instances
            graph: the graph so far in the tracing process
            node: node that represents a conditional
            counter: variable tracking

        Returns: Two sets of constraints. One with a conjunction with the
        the conditional constraint and the other with a conjunction with
        its negation.

        """
        dimension_dict = {}  # type: ignore[var-annotated]

        generator = ConstraintGenerator(tracer_root, graph)
        new_constraints, counter = generator.generate_constraints(counter)

        condition_constraint = new_constraints.conjucts[-1]

        # we know the constraint is a conjunction where the last constraint is about the conditional
        # so remove the last constraint
        new_constraints.conjucts = new_constraints.conjucts[:-1]

        # transform precision, matching, consistency till obtaining a fixed point
        new_constraints, counter = iterate_till_fixed_point(new_constraints, counter)


        # since the function returns a list of one element, we get the first element
        # we are only interested in the RHS in this case because the LHS just stores
        # the result

        # we make sure the constraint is of the form:
        # c = b where b is a boolean expression
        # and we consider b (constraint.rhs) for transformation
        assert isinstance(condition_constraint.lhs, BVar)
        assert is_bool_expr(condition_constraint.rhs)
        condition_constraint_rhs = condition_constraint.rhs

        # transform the condition constraint
        condition_constraint_rhs, counter = iterate_till_fixed_point(condition_constraint_rhs, counter)

        transformed, counter = transform_to_z3(new_constraints, counter, dimension_dict)

        transformed_condition_constraint, counter = transform_to_z3(condition_constraint_rhs, counter, dimension_dict)

        negation_transformed_condition_constraint = z3.Not(transformed_condition_constraint)

        return z3.And([transformed, transformed_condition_constraint]),\
            z3.And([transformed, negation_transformed_condition_constraint])


    def evaluate_conditional_with_constraints(tracer_root, graph, node, counter=0, user_constraints=None):
        """
        Given an IR and a node representing a conditional, evaluate the conditional
        and its negation
        Args:
            tracer_root: Tracer root for module instances
            node: The node to be evaluated

        Returns: the results of evaluating the condition and the negation with
        the rest of the constraints

        """

        transformed_positive, transformed_negative = \
            transform_all_constraints_trace_time(tracer_root, graph, node, counter)

        s = z3.Solver()
        s.add(transformed_positive)
        if user_constraints is not None:
            s.add(user_constraints)
        condition = s.check()

        s = z3.Solver()
        s.add(transformed_negative)
        if user_constraints is not None:
            s.add(user_constraints)
        negation = s.check()
        return condition, negation

except ImportError:
    HAS_Z3 = False
