from typing import Any, List, Optional

from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.attribute.model import Attribute
from torch.testing._internal.inputgen.specs.model import Constraint, ConstraintSuffix
from torch.testing._internal.inputgen.variable.solve import SolvableVariable
from torch.testing._internal.inputgen.variable.type import ScalarDtype


class AttributeSolver:
    def __init__(
        self,
        attribute: Attribute,
        argtype: ArgType,
        scalar_dtype: Optional[ScalarDtype] = None,
    ):
        self.attribute = attribute
        if attribute == Attribute.VALUE and argtype.is_scalar():
            if scalar_dtype is None:
                raise ValueError(
                    "Attribute value for argtype scalar requires a scalar_dtype"
                )
        self.argtype = argtype
        self.vtype = attribute.get_vtype(argtype, scalar_dtype)

    def solve_hard_constraints(self, variable: SolvableVariable) -> None:
        if self.attribute in [Attribute.LENGTH, Attribute.RANK, Attribute.SIZE]:
            variable.Ge(0)

    def solve_user_constraint(
        self,
        variable: SolvableVariable,
        suffix: ConstraintSuffix,
        res: Any,
        valid: bool = True,
    ) -> bool:
        if res is None:
            return False
        if suffix == ConstraintSuffix.EQ:
            variable.Eq(res) if valid else variable.Ne(res)
        if suffix == ConstraintSuffix.NE:
            variable.Ne(res) if valid else variable.Eq(res)
        if suffix == ConstraintSuffix.IN:
            variable.In(res) if valid else variable.NotIn(res)
        if suffix == ConstraintSuffix.NOTIN:
            variable.NotIn(res) if valid else variable.In(res)
        if suffix == ConstraintSuffix.LE:
            variable.Le(res) if valid else variable.Gt(res)
        if suffix == ConstraintSuffix.LT:
            variable.Lt(res) if valid else variable.Ge(res)
        if suffix == ConstraintSuffix.GE:
            variable.Ge(res) if valid else variable.Lt(res)
        if suffix == ConstraintSuffix.GT:
            variable.Gt(res) if valid else variable.Le(res)
        if suffix == ConstraintSuffix.ST:
            variable.St(res) if valid else variable.St(lambda x: not res(x))
        if suffix == ConstraintSuffix.FOCUS:
            if valid:
                variable.In(res)
            else:
                return False
        return True

    def solve_focus_constraints(
        self, variable: SolvableVariable, focus: Attribute
    ) -> None:
        if self.attribute in [Attribute.LENGTH, Attribute.RANK, Attribute.SIZE]:
            if focus in [
                Attribute.LENGTH,
                Attribute.RANK,
                Attribute.SIZE,
                Attribute.VALUE,
            ]:
                attr_pos = Attribute.hierarchy(self.argtype).index(self.attribute)
                focus_pos = Attribute.hierarchy(self.argtype).index(focus)
                if attr_pos < focus_pos:
                    variable.Ge(1)

    def solve(
        self, constraints: List[Constraint], focus: Attribute, valid: bool, *args
    ):
        applicable_constraints = []
        for constraint in constraints:
            if constraint.attribute != self.attribute:
                continue
            res = constraint.fn(*args)
            if res is None:
                continue
            applicable_constraints.append((constraint.suffix, res))

            # TODO(mcandales) This is a hack:
            if constraint.suffix == ConstraintSuffix.GEN:
                valid_values, invalid_values = res
                variable = SolvableVariable(tuple)
                variable.In(valid_values if valid else invalid_values)
                yield variable
                return

        if not valid and self.attribute == focus:
            for invalid_ix in range(len(applicable_constraints)):
                variable = SolvableVariable(self.vtype)
                self.solve_hard_constraints(variable)
                self.solve_focus_constraints(variable, focus)
                for ix, (suffix, res) in enumerate(applicable_constraints):
                    if ix == invalid_ix:
                        if not self.solve_user_constraint(variable, suffix, res, False):
                            break
                    else:
                        self.solve_user_constraint(variable, suffix, res, True)
                else:
                    yield variable
        else:
            variable = SolvableVariable(self.vtype)
            self.solve_hard_constraints(variable)
            self.solve_focus_constraints(variable, focus)
            for suffix, res in applicable_constraints:
                self.solve_user_constraint(variable, suffix, res, True)
            yield variable
