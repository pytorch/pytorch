import torch
from .Module import Module


class SoftPlus(Module):

    def __init__(self, beta=1, threshold=20):
        super(SoftPlus, self).__init__()
        self.beta = beta              # Beta controls sharpness of transfer function
        self.threshold = threshold    # Avoid floating point issues with exp(x), x>20

    def updateOutput(self, input):
        # f(x) = 1/beta * log(1 + exp(beta * x))
        self._backend.SoftPlus_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.beta,
            self.threshold
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        # d/dx[log(1+exp(k*x))/k] = exp(kx) / (exp(kx) + 1)
        # SINCE
        # y = (1/k)*log(1+exp(k*x)) #> x = (1/k)*log(exp(k*y)-1)
        # THEREFORE:
        # d/dx(f(x)) = (exp(k*y) - 1) / exp(k*y)
        self._backend.SoftPlus_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.output,
            self.beta,
            self.threshold
        )
        return self.gradInput
