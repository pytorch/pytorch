import torch

from ..function import Function


class _CompareOp(Function):

    def __init__(self, scalar=None):
        super(_CompareOp, self).__init__()
        self.scalar = scalar

    def forward(self, tensor1, tensor2=None):
        other = tensor2 if tensor2 is not None else self.scalar
        mask = getattr(tensor1, self.fn_name)(other)
        self.mark_non_differentiable(mask)
        return mask


class Eq(_CompareOp):
    fn_name = 'eq'


class Ne(_CompareOp):
    fn_name = 'ne'


class Gt(_CompareOp):
    fn_name = 'gt'


class Ge(_CompareOp):
    fn_name = 'ge'


class Lt(_CompareOp):
    fn_name = 'lt'


class Le(_CompareOp):
    fn_name = 'le'
