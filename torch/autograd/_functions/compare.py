import torch

from ..function import Function
from .utils import maybe_unexpand, maybe_unexpand_or_view


# TODO: once Cpp-style functions are implemented we can detach a and b
# before calling forward.
class _CompareOp(Function):

    @classmethod
    def forward(cls, ctx, a, b):
        ctx.a_size = a.size()
        ctx.b_tensor = torch.is_tensor(b)
        ctx.b_size = b.size() if ctx.b_tensor else None
        ctx.input_type = type(a)
        mask = getattr(a, cls.fn_name)(b)
        ctx.mark_non_differentiable(mask)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = (grad_output * 0).type(ctx.input_type)
        return (maybe_unexpand(grad_input, ctx.a_size),
                maybe_unexpand_or_view(grad_input, ctx.b_size) if ctx.b_tensor else None)


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
