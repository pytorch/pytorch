from ..variable import Variable
from ..function import Function, InplaceFunction

class Exp(InplaceFunction):

    def forward(self, i):
        if self.inplace:
            self.mark_dirty(i)
            result = i.exp_()
        else:
            result = i.exp()
        self.save_for_backward(result)
        return result

    def backward(self, grad_output):
        return self.saved_tensors[0] * grad_output

class Log(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.log()

    def backward(self, grad_output):
        return grad_output.div(self.saved_tensors[0])

class Log1p(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.log1p()

    def backward(self, grad_output):
        return grad_output.div(self.saved_tensors[0].add(1))

