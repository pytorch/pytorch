import random
from typing import Any, List, Optional, Tuple, Union

import torch
from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.attribute.engine import AttributeEngine
from torch.testing._internal.inputgen.attribute.model import Attribute
from torch.testing._internal.inputgen.attribute.solve import AttributeSolver
from torch.testing._internal.inputgen.specs.model import Constraint, ConstraintSuffix
from torch.testing._internal.inputgen.variable.type import ScalarDtype


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

        self.gen_list_mode = set()
        for constraint in constraints:
            if constraint.suffix == ConstraintSuffix.GEN:
                self.gen_list_mode.add(constraint.attribute)

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

        if attr in self.gen_list_mode:
            yield from self.gen_structure_with_depth(depth, focus, length)
            return

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


class MetaArg:
    def __init__(
        self,
        argtype: ArgType,
        *,
        optional: bool = False,
        dtype: Optional[
            Union[torch.dtype, List[Optional[torch.dtype]], ScalarDtype]
        ] = None,
        structure: Optional[Tuple] = None,
        value: Optional[Any] = None,
    ):
        self.argtype = argtype
        self.optional = optional
        self.dtype = dtype
        self.structure = structure
        self.value = value

        if not self.argtype.is_optional() and self.optional:
            raise ValueError("Only optional argtypes can have optional instances")

        if self.argtype.is_tensor_list():
            if len(self.structure) != len(self.dtype):
                raise ValueError(
                    "Structure and dtype must be same length when tensor list"
                )
            if self.argtype == ArgType.TensorList and any(
                d is None for d in self.dtype
            ):
                raise ValueError("Only TensorOptList can have None in list of dtypes")

        if not self.optional and Attribute.DTYPE not in Attribute.hierarchy(
            self.argtype
        ):
            if argtype.is_list():
                self.value = list(self.structure)
            else:
                self.value = self.structure

    def __str__(self):
        if self.optional:
            strval = "None"
        elif self.argtype.is_tensor_list():
            strval = (
                "["
                + ", ".join(
                    [
                        f"{self.dtype[i]} {self.structure[i]}"
                        for i in range(len(self.dtype))
                    ]
                )
                + "]"
            )
        elif self.argtype.is_tensor():
            strval = f"{self.dtype} {self.structure}"
        else:
            strval = str(self.value)
        return f"{self.argtype} {strval}"

    def length(self):
        if self.argtype.is_list():
            return len(self.structure)
        else:
            return None

    def rank(self, ix=None):
        if self.argtype.is_tensor():
            return len(self.structure)
        elif self.argtype.is_tensor_list():
            if ix is None:
                return (len(s) for s in self.structure)
            else:
                return len(self.structure[ix])
        else:
            return None


class MetaArgEngine:
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

    def gen_structures(self, focus):
        if self.argtype.is_scalar():
            yield None
        else:
            yield from StructuralEngine(
                self.argtype, self.constraints, self.deps, self.valid
            ).gen(focus)

    def gen_dtypes(self, focus):
        if not Attribute.DTYPE in Attribute.hierarchy(self.argtype):
            return {None}
        engine = AttributeEngine(
            Attribute.DTYPE, self.constraints, self.valid, self.argtype
        )
        if self.argtype.is_scalar() and focus == Attribute.VALUE:
            # if focused on a scalar value, must generate all dtypes too
            focus = Attribute.DTYPE
        return engine.gen(focus, self.deps)

    def gen_optional(self):
        engine = AttributeEngine(
            Attribute.OPTIONAL, self.constraints, self.valid, self.argtype
        )
        return True in engine.gen(Attribute.OPTIONAL, self.deps)

    def gen_scalars(self, scalar_dtype, focus):
        engine = AttributeEngine(
            Attribute.VALUE, self.constraints, self.valid, self.argtype, scalar_dtype
        )
        return engine.gen(focus, self.deps)

    def gen_value_spaces(self, focus):
        if not self.argtype.is_tensor() and not self.argtype.is_tensor_list():
            return [None]
        solver = AttributeSolver(Attribute.VALUE, self.argtype)
        variables = list(solver.solve(self.constraints, focus, self.valid, self.deps))
        if focus == Attribute.VALUE:
            return [v.space for v in variables]
        else:
            return [random.choice(variables).space]

    def gen(self, focus):
        if self.argtype.is_optional() and self.gen_optional():
            yield MetaArg(self.argtype, optional=True)

        if self.argtype.is_scalar():
            scalar_dtypes = self.gen_dtypes(focus)
            for scalar_dtype in scalar_dtypes:
                for value in self.gen_scalars(scalar_dtype, focus):
                    yield MetaArg(self.argtype, dtype=scalar_dtype, value=value)
        else:
            if focus == Attribute.DTYPE:
                for dtype in self.gen_dtypes(focus):
                    for struct in self.gen_structures(focus):
                        for space in self.gen_value_spaces(focus):
                            yield MetaArg(
                                self.argtype, dtype=dtype, structure=struct, value=space
                            )
            else:
                for struct in self.gen_structures(focus):
                    for dtype in self.gen_dtypes(focus):
                        for space in self.gen_value_spaces(focus):
                            yield MetaArg(
                                self.argtype, dtype=dtype, structure=struct, value=space
                            )
