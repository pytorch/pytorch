# mypy: ignore-errors

import operator
from typing import Dict, List, TYPE_CHECKING

import torch
from torch._dynamo.source import GetItemSource

from .. import variables
from ..exc import unimplemented, UserError, UserErrorType
from ..guards import GuardBuilder, install_guard
from ..utils import common_constant_types, istype, np
from .base import typestr, VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


_type_to_assert_reason = {
    # NB - We CAN have ConstantVariable.create(set) because of how sets interact with guards.
    # A locally created set should always become a SetVariable, as the items in the set will already either be sourced
    # from somewhere else, or unsourced. An input set would imply sources derived from set contents. For example, an
    # input list's contents will have a source like some_list[0], some_list[1][1], etc. For a set, arbitrary access is
    # not possible. This is a solvable problem, but one we have not taken on yet. As such, input sets are not allowed to
    # become SetVariables. The solution here is to create a ConstantSetVariable that is more like a ConstantVariable.
    # As this does not exist, we cannot add sets to this invariant.
    list: "List types must use ListVariable.",
    dict: "Dict types must use ConstDictVariable.",
    torch.Tensor: "Tensor types must use TensorVariable.",
    torch.SymInt: "SymInts must use SymNodeVariable. "
    "If the underlying value is static, we will create a ConstantVariable and specialize.",
    torch.SymFloat: "SymInts must use SymNodeVariable",
}


class ConstantVariable(VariableTracker):
    @staticmethod
    def create(value, **kwargs) -> VariableTracker:
        source = kwargs.get("source", None)
        is_literal = ConstantVariable.is_literal(value)
        if not is_literal:
            for disallowed_type, reason in _type_to_assert_reason.items():
                assert not isinstance(value, disallowed_type), reason

        # Routing for list and tuple literals.
        if is_literal and isinstance(value, (set, frozenset)):
            items = []
            for i, x in enumerate(value):
                items.append(ConstantVariable.create(x))
            return variables.SetVariable(items, **kwargs)
        elif is_literal and isinstance(value, (list, tuple)):
            items = []
            for i, x in enumerate(value):
                item_source = GetItemSource(source, i) if source else None
                if item_source:
                    install_guard(item_source.make_guard(GuardBuilder.CONSTANT_MATCH))
                items.append(
                    ConstantVariable.create(
                        x,
                        source=item_source,
                    )
                )
            return variables.BaseListVariable.cls_for(type(value))(items, **kwargs)

        return ConstantVariable(value, **kwargs)

    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        if not ConstantVariable.is_literal(value):
            for disallowed_type, reason in _type_to_assert_reason.items():
                assert not isinstance(value, disallowed_type), reason

        assert not isinstance(
            value, (list, tuple)
        ), "ConstantVariable(list) is banned - please create a ListVariable(items)"
        if np is not None and isinstance(value, np.number):
            self.value = value.item()
        else:
            self.value = value

    def as_proxy(self):
        return self.value

    def __str__(self) -> str:
        return f"ConstantVariable({type(self.value).__name__}: {repr(self.value)})"

    def as_python_constant(self):
        return self.value

    def is_python_constant(self):
        return True

    @property
    def items(self):
        """
        Need this when adding a BaseListVariable and a ConstantVariable together.
        Happens in detectron2.
        """
        return self.unpack_var_sequence(tx=None)

    def getitem_const(self, tx: "InstructionTranslator", arg: VariableTracker):
        return ConstantVariable.create(
            self.value[arg.as_python_constant()],
        )

    @staticmethod
    def is_literal(obj):
        if type(obj) in common_constant_types:
            return True
        # The structure within is_literal get routed to variables.BaseListVariable
        if type(obj) in (list, tuple, set, frozenset, torch.Size):
            return all(ConstantVariable.is_literal(x) for x in obj)
        return False

    def unpack_var_sequence(self, tx):
        try:
            return [ConstantVariable.create(x) for x in self.as_python_constant()]
        except TypeError as e:
            raise NotImplementedError from e

    def const_getattr(self, tx: "InstructionTranslator", name):
        if isinstance(self.value, type):
            raise UserError(
                UserErrorType.ANTI_PATTERN,
                "Can't access members of type(obj) for a generated custom object. "
                "Please use __class__ instead",
                case_name="type_reflection_method",
            )
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError
        return member

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .tensor import SymNodeVariable

        if name == "format" and istype(self.value, str):
            return variables.BuiltinVariable(str.format).call_function(
                tx, [self, *args], kwargs
            )

        if any(isinstance(x, SymNodeVariable) for x in args):
            # Promote to SymNodeVariable for operations involving dynamic shapes.
            return variables.SymNodeVariable(self.as_proxy(), self.value).call_method(
                tx, name, args, kwargs
            )

        try:
            const_args = [a.as_python_constant() for a in args]
            const_kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
        except NotImplementedError:
            return super().call_method(tx, name, args, kwargs)

        if isinstance(self.value, str) and name in str.__dict__.keys():
            method = getattr(self.value, name)
            return ConstantVariable.create(method(*const_args, **const_kwargs))
        elif isinstance(self.value, (float, int)):
            if not (args or kwargs):
                return ConstantVariable.create(getattr(self.value, name)())
            if (
                hasattr(operator, name)
                and len(args) == 1
                and args[0].is_python_constant()
            ):
                add_target = const_args[0]
                op = getattr(operator, name)
                if isinstance(
                    add_target, (torch.SymBool, torch.SymFloat, torch.SymInt)
                ):
                    # Addition between a non sym and sym makes a sym
                    proxy = tx.output.create_proxy(
                        "call_function", op, (self.value, add_target), {}
                    )
                    return SymNodeVariable.create(tx, proxy, add_target)
                else:
                    return ConstantVariable.create(op(self.value, add_target))
        elif isinstance(self.value, bytes) and name == "decode":
            method = getattr(self.value, name)
            return ConstantVariable.create(method(*const_args, **const_kwargs))

        if name == "__len__" and not (args or kwargs):
            return ConstantVariable.create(len(self.value))
        elif name == "__contains__" and len(args) == 1 and args[0].is_python_constant():
            assert not kwargs
            search = args[0].as_python_constant()
            result = search in self.value
            return ConstantVariable.create(result)

        unimplemented(f"const method call {typestr(self.value)}.{name}")

    def call_hasattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        result = hasattr(self.value, name)
        return variables.ConstantVariable.create(result)


class EnumVariable(VariableTracker):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    @classmethod
    def create(cls, cls_type, value_vt, options):
        if isinstance(value_vt, variables.ConstantVariable):
            for member in list(cls_type):
                if member.value == value_vt.as_python_constant():
                    return cls(member, **options)
        unimplemented("Enum variable is constructed with non constant values")

    def as_proxy(self):
        return self.value

    def __str__(self) -> str:
        return f"EnumVariable({type(self.value)})"

    def as_python_constant(self):
        return self.value

    def const_getattr(self, tx: "InstructionTranslator", name):
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError
        return member
