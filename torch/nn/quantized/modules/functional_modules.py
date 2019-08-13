import torch
from torch._ops import ops


class FloatFunctional(torch.nn.Module):
    r"""State collector class for float operatitons.

    The instance of this class can be used instead of the ``torch.`` prefix for
    some operations. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    .. Examples::

        >>> f_add = FloatFunctional()
        >>> a = torch.tensor(3.0)
        >>> b = torch.tensor(4.0)
        >>> f_add.add(a, b)  # Equivalent to ``torch.add(3, 4)

    Valid operation names:
        - add
        - cat
    """
    def __init__(self):
        super(FloatFunctional, self).__init__()
        self.observer = torch.nn.Identity()

    def forward(self, x):
        raise RuntimeError("FloatFunctional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    r"""Operation equivalent to ``torch.add``"""
    def add(self, x, y):
        # type: (Tensor, Tensor) -> Tensor
        r = torch.add(x, y)
        # TODO: Fix for QAT.
        self.observer(r)
        return r

    r"""Operation equivalent to ``torch.cat``"""
    def cat(self, x, dim=0):
        # type: (List[Tensor], int) -> Tensor
        r = torch.cat(x, dim=dim)
        self.observer(r)
        return r


class QFunctional(torch.nn.Module):
    r"""Wrapper class for quantized operatitons.

    The instance of this class can be used instead of the
    ``torch.ops.quantized`` prefix. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    .. Examples::

        >>> q_add = QFunctional('add')
        >>> a = torch.quantize_linear(torch.tensor(3.0), 1.0, 0, torch.qint32)
        >>> b = torch.quantize_linear(torch.tensor(4.0), 1.0, 0, torch.qint32)
        >>> q_add.add(a, b)  # Equivalent to ``torch.ops.quantized.add(3, 4)

    Valid operation names:
        - add
        - cat

    """
    def __init__(self):
        super(QFunctional, self).__init__()

    def forward(self, x):
        raise RuntimeError("Functional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    r"""Operation equivalent to ``torch.ops.quantized.add``"""
    def add(self, x, y):
        return ops.quantized.add(x, y, scale=self.scale,
                                 zero_point=self.zero_point)

    r"""Operation equivalent to ``torch.ops.quantized.cat``"""
    def cat(self, x, dim=0):
        # type: (List[Tensor], int) -> Tensor
        return ops.quantized.cat(x, scale=self.scale,
                                 zero_point=self.zero_point, dim=dim)

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == FloatFunctional,\
            "QFunctional.from_float expects an instance of FloatFunctional"
        scale, zero_point = mod.observer.calculate_qparams()[:2]
        new_mod = QFunctional()
        new_mod.scale = scale
        new_mod.zero_point = zero_point
        return new_mod
