import random
from typing import Any, List, Optional

from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.attribute.engine import AttributeEngine
from torch.testing._internal.inputgen.attribute.model import Attribute
from torch.testing._internal.inputgen.specs.model import Constraint


class StructuralEngine:
    def __init__(
        self,
        argtype: ArgType,
        constraints: List[Constraint],
        deps: List[Any],
        valid: bool,
    ):
        self.argtype = argtype
        self.constraints = constraints
        self.deps = deps
        self.valid = valid
        self.hierarchy = StructuralEngine.hierarchy(argtype)

    @staticmethod
    def hierarchy(argtype) -> List[Attribute]:
        """Return the structural hierarchy for a given argument type"""
        if argtype.is_tensor_list():
            return [Attribute.LENGTH, Attribute.RANK, Attribute.SIZE]
        elif argtype.is_tensor():
            return [Attribute.RANK, Attribute.SIZE]
        elif argtype.is_list():
            return [Attribute.LENGTH, Attribute.VALUE]
        else:
            return [Attribute.VALUE]

    def gen_structure_with_depth_and_length(
        self, depth: int, length: int, focus: Attribute
    ):
        if length == 0:
            yield ()
            return

        attr = self.hierarchy[-(depth + 1)]
        focus_ixs = range(length) if focus == attr else (random.choice(range(length)),)
        for focus_ix in focus_ixs:
            values = [()]
            for ix in range(length):
                if ix == focus_ix:
                    elements = self.gen_structure_with_depth(depth, focus, length, ix)
                else:
                    elements = self.gen_structure_with_depth(depth, None, length, ix)
                new_values = []
                for elem in elements:
                    new_values += [t + (elem,) for t in values]
                values = new_values
            yield from values

    def gen_structure_with_depth(
        self,
        depth: int,
        focus: Attribute,
        length: Optional[int] = None,
        ix: Optional[int] = None,
    ):
        attr = self.hierarchy[-(depth + 1)]

        if ix is not None:
            args = (self.deps, length, ix)
        elif length is not None:
            args = (
                self.deps,
                length,
            )
        else:
            args = (self.deps,)

        values = AttributeEngine(attr, self.constraints, self.valid, self.argtype).gen(
            focus, *args
        )

        for v in values:
            if depth == 0:
                yield v
            else:
                yield from self.gen_structure_with_depth_and_length(depth - 1, v, focus)

    def gen(self, focus: Attribute):
        depth = len(self.hierarchy) - 1
        yield from self.gen_structure_with_depth(depth, focus)
