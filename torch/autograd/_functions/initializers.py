from ..function import InplaceFunction


class Zero(InplaceFunction):

    @staticmethod
    def forward(ctx, i, inplace=False):
        if inplace:
            ctx.mark_dirty(i)
            result = i.zero_()
        else:
            result = i.new(1).zero_().expand_as(i)

        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_variables
        return Variable(result.data.new(1).zero_().expand_as(result))
