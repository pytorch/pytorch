# mypy: ignore-errors

import operator
from typing import Dict, List, TYPE_CHECKING

import torch
from torch._dynamo.source import AttrSource, GetItemSource

from .. import variables
from ..exc import unimplemented
from ..utils import common_constant_types, istype, np
from .base import typestr, VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


class ConstantVariable(VariableTracker):
    @staticmethod
    def create(value, **kwargs) -> VariableTracker:
        """
        Create a `ConstantVariable` based on the given value, and supports
        automatic routing for collection types like `tuple` (in which case we'd
        create `ConstantVariable` for the leaf items).

        NOTE: the caller must install the proper guards if needed; most often
        the guard will be `CONSTANT_MATCH`.
        """
        source = kwargs.get("source", None)

        # Routing for supported collection literals.
        if isinstance(value, set):
            items = [ConstantVariable.create(x) for x in value]
            return variables.SetVariable(items, **kwargs)
        elif isinstance(value, frozenset):
            items = [ConstantVariable.create(x) for x in value]
            return variables.FrozensetVariable(items, **kwargs)
        elif isinstance(value, (list, tuple)):
            items = []
            for i, x in enumerate(value):
                item_source = GetItemSource(source, i) if source else None
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
        assert ConstantVariable.is_base_literal(
            value
        ), f"""
Cannot construct `ConstantVariable` for value of type {type(value)}.

This failure likely due to PyTorch-internal use of `ConstantVariable` on
non-literal python values, please try using `VariableTracker.build` instead. If
you believe it's a necessary and legitimate use case (the value is immutable and
can't easily be represented with another `VariableTracker` class), please add
its type to `common_constant_types`.
"""
        if np is not None and isinstance(value, np.number):
            self.value = value.item()
        else:
            self.value = value

    def as_proxy(self):
        return self.value

    def __repr__(self) -> str:
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
    def is_base_literal(obj):
        return type(obj) in common_constant_types

    @staticmethod
    def is_literal(obj):
        if type(obj) in (list, tuple, set, frozenset, torch.Size):
            return all(ConstantVariable.is_literal(x) for x in obj)
        return ConstantVariable.is_base_literal(obj)

    def unpack_var_sequence(self, tx):
        try:
            return [ConstantVariable.create(x) for x in self.as_python_constant()]
        except TypeError as e:
            raise NotImplementedError from e

    def const_getattr(self, tx: "InstructionTranslator", name):
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
        elif name == "join" and istype(self.value, str):
            assert len(args) == 1 and len(kwargs) == 0
            arg_unpacked = args[0].force_unpack_var_sequence(tx)
            try:
                arg_const = [x.as_python_constant() for x in arg_unpacked]
                return ConstantVariable.create(self.value.join(arg_const))
            except NotImplementedError:
                return super().call_method(tx, name, args, kwargs)

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
        elif name == "__round__" and len(args) == 1 and args[0].is_python_constant():
            return ConstantVariable.create(
                round(self.value, args[0].is_python_constant())
            )
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
        if isinstance(self.value, int):
            return int(self.value)  # convert IntEnum to a normal int
        return self.value

    def __repr__(self) -> str:
        return f"EnumVariable({type(self.value)})"

    def as_python_constant(self):
        return self.value

    def var_getattr(self, tx: "InstructionTranslator", name):
        member = getattr(self.value, name)
        source = self.source and AttrSource(self.source, name)
        return VariableTracker.build(tx, member, source=source)
