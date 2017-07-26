from ..function import InplaceFunction


class Zero(InplaceFunction):

    @staticmethod
    def forward(ctx, i, inplace=False):
        if inplace:
            ctx.mark_dirty(i)
            result = i.zero_()
        else:
            result = i.new(i.size()).zero_()

        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_variables
        return result.new(result.size()).zero_()
