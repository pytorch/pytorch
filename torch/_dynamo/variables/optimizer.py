from .user_defined import UserDefinedObjectVariable
from typing import List, Dict
from .base import VariableTracker
from .constant import ConstantVariable
from .misc import GetAttrVariable



class OptimizerVariable(UserDefinedObjectVariable):
    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        """This is an optimization to avoid tracing the very slow intialization of the optimizer"""
        if name == "_init_group":
            self.value._init_group(*args, **kwargs)
            return ConstantVariable(None)
        else:
            super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name):
        if name == "_init_group":
            return GetAttrVariable(self, name)

        return super().var_getattr(tx, name)
