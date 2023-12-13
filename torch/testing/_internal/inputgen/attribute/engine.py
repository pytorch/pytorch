import copy
from typing import List, Optional

from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.attribute.model import Attribute
from torch.testing._internal.inputgen.attribute.solve import AttributeSolver
from torch.testing._internal.inputgen.specs.model import Constraint
from torch.testing._internal.inputgen.variable.gen import VariableGenerator
from torch.testing._internal.inputgen.variable.type import ScalarDtype


class AttributeEngine(AttributeSolver):
    def __init__(
        self,
        attribute: Attribute,
        constraints: List[Constraint],
        valid: bool,
        argtype: Optional[ArgType] = None,
        scalar_dtype: Optional[ScalarDtype] = None,
    ):
        super().__init__(attribute, argtype, scalar_dtype)
        self.constraints = constraints
        self.valid = valid

    def gen(self, focus: Attribute, *args):
        if self.attribute == Attribute.OPTIONAL:
            num = 2
        elif self.attribute == focus:
            if self.attribute == Attribute.DTYPE:
                num = 8
            else:
                num = 6
        else:
            num = 1
        gen_vals = set()
        for variable in self.solve(self.constraints, focus, self.valid, *args):
            vals = []
            if variable.vtype in [bool, int, float]:
                limits = self.attribute.get_custom_limits(self.argtype)
                if limits is not None:
                    v_copy = copy.deepcopy(variable)
                    v_copy.Ge(limits[0])
                    v_copy.Le(limits[1])
                    vals = VariableGenerator(v_copy.space).gen(num)
            if len(vals) == 0:
                vals = VariableGenerator(variable.space).gen(num)
            gen_vals.update(vals)
        return gen_vals
