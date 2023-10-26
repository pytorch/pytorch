from typing import Dict, List

from .base import MutableLocal, VariableTracker
from .lists import TupleVariable
from .tensor import _register_hook
from .user_defined import UserDefinedObjectVariable


class AccumulateGradVariable(VariableTracker):
    def __init__(self, value, proxy, value_type=None, **kwargs):
        self.proxy = proxy
        self.value = value
        self.value_type = value_type
        super().__init__(**kwargs)

    def call_method(
        self, tx, name, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]
    ) -> VariableTracker:
        if name == "register_hook":
            return _register_hook(self, tx, args, kwargs)
        return super().call_method(tx, name, args, kwargs)

    def as_proxy(self):
        return self.proxy


class AutogradNodeVariable(UserDefinedObjectVariable):
    def __init__(self, value, proxy, value_type=None, **kwargs):
        self.proxy = proxy
        super().__init__(value, value_type, **kwargs)

    def var_getattr(self, tx, name):
        attr = getattr(self.value, name, None)
        # print(f"Asking for {name} on node and it has? {attr} {self.source}")
        options = VariableTracker.propagate(self)
        if attr and self.source:
            from .builder import VariableBuilder

            return VariableBuilder(tx, AttrSource(self.source, name))(
                getattr(self.value, name)
            )
        elif attr and name == "next_functions":
            outer_tuple_items = []
            for i, outer_item in enumerate(attr):
                inner_tuple_items = []
                for j, inner_item in enumerate(outer_item):
                    options["mutable_local"] = MutableLocal()
                    acc_grad_proxy = self.proxy.next_functions[i][j]
                    acc_grad_proxy.node.meta["example_value"] = inner_item
                    inner_tuple_items.append(
                        AccumulateGradVariable(
                            inner_item, acc_grad_proxy, **options
                        )
                    )
                inner_tuple_obj = TupleVariable(inner_tuple_items, **options)
                outer_tuple_items.append(inner_tuple_obj)
            outer_tuple_obj = TupleVariable(outer_tuple_items, **options)
            result = outer_tuple_obj
            return result
        return super().var_getattr(tx, name)
