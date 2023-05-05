from torch._ops import HigherOrderOperator

# Used for testing the HigherOrderOperator mechanism
class Wrap(HigherOrderOperator):
    def __init__(self):
        super().__init__("wrap")

    def __call__(self, func, *args):
        result = func(*args)
        return result


wrap = Wrap()
