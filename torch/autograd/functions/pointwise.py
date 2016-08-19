from ..variable import Variable
from ..function import Function

class Exp(Function):

    def forward(self, i):
        self.result = i.exp()
        return self.result

    def backward(self, grad_output):
        return self.result * grad_output

class Log(Function):

    def forward(self, i):
        self.input = i
        return i.log()

    def backward(self, grad_output):
        return grad_output.div(self.input)

class Log1p(Function):

    def forward(self, i):
        self.input = i
        return i.log1p()

    def backward(self, grad_output):
        return grad_output.div(self.input.add(1))

