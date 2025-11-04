import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator


class Print(HigherOrderOperator):
    def __init__(self):
        super().__init__("hop_print")
        self._print_str = None

    def __call__(self, *args, **kwargs):
        if self._print_str is None:
            self._print_str = self._get_print_str()
        print(self._print_str)
        return self._fn(*args, **kwargs)

    def _get_print_str(self):
        return


print = Print()
