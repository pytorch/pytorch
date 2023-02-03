from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]

from ._compatibility import compatibility


@compatibility(is_backward_compatible=False)
class TensorType:
    """
    TensorType defines a type for tensors, which consists of a list of dimensions.
    Example:
        class M(torch.nn.Module):
            def forward(self, x:TensorType((1,2,3, Dyn)), y:TensorType((1,2,3, Dyn))):
                return torch.add(x, y)
    """

    def __init__(self, dim):
        self.__origin__ = TensorType
        self.__args__ = dim

    def __repr__(self):
        return f'TensorType[{self.__args__}]'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return list(self.__args__) == list(other.__args__)
        else:
            return False

    @staticmethod
    def __class_getitem__(*args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return TensorType(tuple(args))


class _DynType:
    """
    _DynType defines a type which stands for the absence of type information.
    """
    def __init__(self):
        self.__name__ = '_DynType'

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __str__(self):
        return "Dyn"

    def __repr__(self):
        return "Dyn"


Dyn = _DynType()

@compatibility(is_backward_compatible=False)
def is_consistent(t1, t2):
    """
    A binary relation denoted by ~ that determines if t1 is consistent with t2.
    The relation is reflexive, semmetric but not transitive.
    returns True if t1 and t2 are consistent and False otherwise.
    Example:
        Dyn ~ TensorType((1,2,3))
        int ~ Dyn
        int ~ int
        TensorType((1,Dyn,3)) ~ TensorType((1,2,3))
    """

    if t1 == t2:
        return True

    if t1 == Dyn or t2 == Dyn or isinstance(t1, Var) or isinstance(t2, Var):
        return True

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.__args__) == len(t2.__args__) and \
            all([is_consistent(elem1, elem2) for elem1, elem2 in zip(t1.__args__, t2.__args__)])
    else:
        return False


@compatibility(is_backward_compatible=False)
def is_more_precise(t1, t2):
    """
    A binary relation denoted by <= that determines if t1 is more precise than t2.
    The relation is reflexive and transitive.
    returns True if t1 is more precise than t2 and False otherwise.
    Example:
        Dyn >= TensorType((1,2,3))
        int >= Dyn
        int >= int
        TensorType((1,Dyn,3)) <= TensorType((1,2,3))
    """
    if t1 == t2:
        return True

    if isinstance(t2, _DynType):
        return True

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.__args__) == len(t2.__args__) and \
            all([is_more_precise(elem1, elem2) for elem1, elem2 in zip(t1.__args__, t2.__args__)])

    else:
        return False
