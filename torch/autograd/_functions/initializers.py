from ..function import InplaceFunction
from .tensor import NoGrad

class Zero(InplaceFunction):

    @staticmethod
    def forward(ctx, i, inplace=False):
        if inplace:
            ctx.mark_dirty(i)
            result = i.zero_()
        else:
            result = i.new(i.size()).zero_()
        ctx.mark_non_differentiable(result)
        return result

