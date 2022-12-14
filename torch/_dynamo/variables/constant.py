import operator
from typing import Dict, List

import torch

from .. import variables
from ..exc import unimplemented
from ..utils import istype
from .base import typestr, VariableTracker


class ConstantVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super(ConstantVariable, self).__init__(**kwargs)
        assert not isinstance(value, torch.Tensor)
        assert not isinstance(value, torch.SymInt)
        assert not isinstance(value, torch.SymFloat)
        self.value = value

    def as_proxy(self):
        return self.value

    def __str__(self):
        # return f"ConstantVariable({self.value})"
        return f"ConstantVariable({type(self.value).__name__})"

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    @property
    def items(self):
        """
        Need this when adding a BaseListVariable and a ConstantVariable together.
        Happens in detectron2.
        """
        return self.unpack_var_sequence(tx=None)

    def getitem_const(self, arg: VariableTracker):
        return ConstantVariable(
            self.value[arg.as_python_constant()],
            **VariableTracker.propagate([self, arg]),
        )

    @staticmethod
    def is_literal(obj):
        if type(obj) in (int, float, bool, type(None), str):
            return True
        if type(obj) in (list, tuple, set, frozenset):
            return all(ConstantVariable.is_literal(x) for x in obj)
        return False

    def unpack_var_sequence(self, tx):
        try:
            options = VariableTracker.propagate([self])
            return [ConstantVariable(x, **options) for x in self.as_python_constant()]
        except TypeError:
            raise NotImplementedError()

    def const_getattr(self, tx, name):
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError()
        return member

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .tensor import DynamicShapeVariable

        options = VariableTracker.propagate(self, args, kwargs.values())

        if istype(self.value, tuple):
            # empty tuple constant etc
            return variables.TupleVariable(
                items=self.unpack_var_sequence(tx), source=self.source, **options
            ).call_method(tx, name, args, kwargs)

        if any([isinstance(x, DynamicShapeVariable) for x in args]):
            # NOTE! DANGER! THIS ONLY WORKS FOR COMMUTATIVE OPS
            # we are relying on add to have arg[0] be a DynamicShapeVariable
            # because we are in ConstantVariable land
            # This transforms
            # constant + dynamic
            # into
            # dynamic + constant
            # Which already has infra built for writing to the graph
            if name == "__add__":
                assert len(args) == 1
                return args[0].call_method(tx, name, [self], {})
            # Unfortunate constant
            return super(ConstantVariable, self).call_method(tx, name, args, kwargs)
        try:
            const_args = [a.as_python_constant() for a in args]
            const_kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
        except NotImplementedError:
            return super(ConstantVariable, self).call_method(tx, name, args, kwargs)

        def has_arith_binop(num_ty):
            return (
                isinstance(self.value, num_ty)
                and hasattr(operator, name)
                and len(args) == 1
                and args[0].is_python_constant()
            )

        if isinstance(self.value, str) and name in str.__dict__.keys():
            assert not kwargs
            method = getattr(self.value, name)
            return ConstantVariable(method(*const_args, **const_kwargs), **options)
        elif has_arith_binop(int) or has_arith_binop(float):
            op = getattr(operator, name)
            add_target = const_args[0]
            if isinstance(add_target, (torch.SymInt, torch.SymFloat)):
                from .tensor import DynamicShapeVariable

                # Addition between a non sym and sym makes a sym
                # dyn_shape = tx.output.register_attr_or_module(
                #     add_target, f"sym_shape_{add_target}", source=None
                # )
                proxy = tx.output.create_proxy(
                    "call_function", op, (self.value, add_target), {}
                )
                return DynamicShapeVariable.create(tx, proxy, add_target, **options)
            return ConstantVariable(op(self.value, add_target), **options)
        elif name == "__len__" and not (args or kwargs):
            return ConstantVariable(len(self.value), **options)
        elif name == "__contains__" and len(args) == 1 and args[0].is_python_constant():
            assert not kwargs
            search = args[0].as_python_constant()
            result = search in self.value
            return ConstantVariable(result, **options)

        unimplemented(f"const method call {typestr(self.value)}.{name}")


class EnumVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super(EnumVariable, self).__init__(**kwargs)
        self.value = value

    def as_proxy(self):
        return self.value

    def __str__(self):
        return f"EnumVariable({type(self.value)})"

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value
