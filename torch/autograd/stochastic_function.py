from .function import Function

_NOT_PROVIDED = object()


class StochasticFunction(Function):

    def __init__(self):
        self.reward = _NOT_PROVIDED

    def _do_backward(self, grad_output, retain_variables):
        if self.reward is _NOT_PROVIDED:
            raise RuntimeError("differentiating stochastic functions requires "
                               "providing a reward")
        result = super(StochasticFunction, self)._do_backward((self.reward,), retain_variables)
        if not retain_variables:
            self.reward = None
        return result

    def _reinforce(self, reward):
        self.reward = reward
