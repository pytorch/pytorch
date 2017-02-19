import torch
from numbers import Number
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

    def _do_forward(self, *inputs):
        result = super(StochasticFunction, self)._do_forward(*inputs)
        # save output type and size, to check the type of reward
        assert isinstance(result, torch.autograd.Variable), \
            "stochastic functions support only a single output at the moment"
        self.reward_info = (type(inputs[0].data), result.size())
        return result

    __call__ = _do_forward

    def _reinforce(self, reward):
        is_number = isinstance(reward, Number)
        if not is_number and type(reward) != self.reward_info[0]:
            raise TypeError("mismatch between reward and output type: got {}, "
                            "but expected {}".format(torch.typename(reward),
                                                     torch.typename(self.reward_info[0])))
        if not is_number and reward.size() != self.reward_info[1]:
            raise ValueError("got reward of size {}, but expected a tensor of size {}".format(
                             'x'.join(map(str, reward.size())),
                             'x'.join(map(str, self.reward_info[1]))))
        if self.reward is not _NOT_PROVIDED:
            raise RuntimeError("you can only reinforce a stochastic Function once")
        self.reward = reward
