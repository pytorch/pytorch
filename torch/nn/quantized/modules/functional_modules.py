import torch
from torch._ops import ops


_VALID_OP_NAMES = {
    'add',
    'cat'
}

class FloatFunctional(torch.nn.Module):
    def __init__(self, op_name):
        super(FloatFunctional, self).__init__()
        if op_name not in _VALID_OP_NAMES:
            raise NotImplementedError("{} not supported.".format(op_name))
        self.observer = torch.nn.Identity()
        self.op_name = op_name

    def forward(self, x):
        raise RuntimeError("FloatFunctional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    def add(self, x, y):
        # type: (Tensor, Tensor) -> Tensor
        assert self.op_name == 'add', \
            "Running add while initialized as {}".format(self.op_name)
        r = torch.add(x, y)
        self.observer(r)
        return r

    def cat(self, x, dim=None):
        # type: (List[Tensor], Optional[int]) -> Tensor
        assert self.op_name == 'cat', \
            "Running cat while initialized as {}".format(self.op_name)
        if dim is None:
            dim = 0
        r = torch.cat(x, dim=dim)
        self.observer(r)
        return r


class QFunctional(torch.nn.Module):
    def __init__(self, op_name):
        super(QFunctional, self).__init__()
        if op_name not in _VALID_OP_NAMES:
            raise NotImplementedError("{} not supported.".format(op_name))
        self.op_name = op_name

    def forward(self, x):
        raise RuntimeError("WFunctional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    def add(self, x, y):
        # type: (Tensor, Tensor) -> Tensor
        assert self.op_name == 'add', \
            "Running add while initialized as {}".format(self.op_name)
        return ops.quantized.add(x, y, scale=self.scale,
                                 zero_point=self.zero_point)

    def cat(self, x, dim=None):
        # type: (List[Tensor], Optional[int]) -> Tensor
        assert self.op_name == 'cat', \
            "Running cat while initialized as {}".format(self.op_name)
        if dim is None:
            dim = 0
        return ops.quantized.cat(x, scale=self.scale,
                                 zero_point=self.zero_point, dim=dim)

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == FloatFunctional,\
            "QFunctional.from_float expects an instance of FloatFunctional"
        scale, zero_point = mod.observer.calculate_qparams()[:2]
        new_mod = QFunctional(mod.op_name)
        new_mod.scale = scale
        new_mod.zero_point = zero_point
        return new_mod
