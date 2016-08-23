from torch.autograd import Variable
from .module import Module

def _assert_no_grad(variable):
    assert not variable.creator.requires_grad, "nn criterions don't compute " \
        "the gradient w.r.t. targets - please mark these variables as not" \
        "requiring gradients"

class AbsCriterion(Module):

    def __init__(self, size_average=True):
        super(AbsCriterion, self).__init__()
        self.size_average = size_average

    def _forward(self, input, target):
        if isinstance(target, Variable):
            _assert_no_grad(target)
            target = target.data
        return self._backend.AbsCriterion(target, self.size_average)(input)

class ClassNLLCriterion(Module):

    def __init__(self, weight=None, size_average=True):
        super(ClassNLLCriterion, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def _forward(self, input, target):
        if isinstance(target, Variable):
            _assert_no_grad(target)
            target = target.data
        return self._backend.ClassNLLCriterion(target, self.size_average, weight=self.weight)(input)

