import torch

class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        result = result.log()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

def fn(input):
    result = input + 5
    return Exp.apply(result) + 3

input = torch.ones(1)
traced_exp = torch.jit.trace(fn, input)
print(traced_exp.graph)

torch.onnx.export(traced_exp, input, 'model.onnx')