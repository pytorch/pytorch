import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator


# This is a functional version of torch.print to enable format printing
# through calling such as
# torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
class Print(HigherOrderOperator):
    def __init__(self):
        super().__init__("print")
        self._print_str = None

    def __call__(self, format_str, **kwargs):
        self._print_str = format_str
        self._print_kwargs = kwargs
        return super().__call__(format_str, **kwargs)

    def _get_print_str(self):
        return self._print_str


print = Print()


@print.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
# pyre-ignore
def print_cpu(format_str, **kwargs):
    # Ensure all immutable_dict/list in kwargs are converted to regular dict/list
    map_types: dict[type, type] = {
        torch.fx.immutable_collections.immutable_dict: dict,
        torch.fx.immutable_collections.immutable_list: list,
    }
    if kwargs:
        new_kwargs = pytree.tree_map_only(
            tuple(map_types.keys()),
            lambda a: map_types[type(a)](a),
            kwargs,
            lambda a: isinstance(a, tuple(map_types.keys())),
        )
        print_str = format_str.format(**new_kwargs)
    else:
        print_str = format_str
    # Use built-in print to avoid recursion with the HOP print
    __builtins__["print"](print_str)
    return None
