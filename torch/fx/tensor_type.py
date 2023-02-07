from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]

from ._compatibility import compatibility


@compatibility(is_backward_compatible=False)
class TensorType:
    """
    TensorType defines a type for tensors, which consists of a list of dimensions.
    Example:
        class M(torch.nn.Module):
            def forward(self, x:TensorType(Dyn, 2 , 3, Static), y:TensorType(Dyn, 2, 3, Static)):
                return torch.add(x, y)
    """

    def __init__(self, *dims):
        self.__origin__ = TensorType

        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            self.__args__ = dims[0]
        else:
            self.__args__ = dims

    def __len__(self):
        return len(self.__args__)

    def __repr__(self):
        return f'TensorType{self.__args__}'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return list(self.__args__) == list(other.__args__)
        else:
            return False

    def __getitem__(self, key):
        return self.__args__[key]

    @staticmethod
    def __class_getitem__(*args):
        return TensorType(*args)


class _DynDim:
    """
    _DynDim represent a dimension with dynamic size
    """
    def __init__(self):
        self.__name__ = '_DynDim'

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __str__(self):
        return "Dyn"

    def __repr__(self):
        return "Dyn"


Dyn = _DynDim()

class _StaticDim:
    """
    _StaticDim represent a dimension with static size
    """
    def __init__(self):
        self.__name__ = '_StaticDim'

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __str__(self):
        return "Static"

    def __repr__(self):
        return "Static"

Static = _StaticDim()


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
        return len(t1) == len(t2) and \
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

    if isinstance(t2, _DynDim):
        return True

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1) == len(t2) and \
            all([is_more_precise(elem1, elem2) for elem1, elem2 in zip(t1.__args__, t2.__args__)])

    else:
        return False
