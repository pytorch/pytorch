from torch.autograd import Variable
from .module import Module
from .container import Sequential
from .activation import LogSoftmax

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(Module):

    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        _assert_no_grad(target)
        backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(self.size_average)(input, target)


class _WeighedLoss(_Loss):

    def __init__(self, weight=None, size_average=True):
        super(_WeighedLoss, self).__init__(size_average)
        self.weight = weight

    def forward(self, input, target):
        _assert_no_grad(target)
        backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(self.size_average, weight=self.weight)(input, target)


class L1Loss(_Loss):
    pass


class NLLLoss(_WeighedLoss):
    pass


class NLLLoss2d(_Loss):
    pass


class KLDivLoss(_WeighedLoss):
    pass


class MSELoss(_Loss):
    pass


class BCELoss(_WeighedLoss):
    pass


class HingeEmbeddingLoss(_Loss):
    pass


class MultiLabelMarginLoss(_Loss):
    pass


class SmoothL1Loss(_Loss):
    pass


class SoftMarginLoss(_Loss):
    pass


class CrossEntropyLoss(_WeighedLoss):

    def forward(self, input, target):
        _assert_no_grad(target)
        log = self._backend.LogSoftmax()(input)
        return self._backend.NLLLoss(self.size_average,
                weight=self.weight)(log, target)


class MultiLabelSoftMarginLoss(_WeighedLoss):

    def forward(self, input, target):
        sigmoid = self._backend.Sigmoid()(input)
        return self._backend.BCELoss(self.size_average, weight=self.weight)(
                sigmoid, target)


class CosineEmbeddingLoss(Module):

    def __init__(self, margin=0, size_average=True):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return self._backend.CosineEmbeddingLoss(self.margin,
                self.size_average)(input1, input2, target)


class MarginRankingLoss(Module):

    def __init__(self, margin=0, size_average=True):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return self._backend.MarginRankingLoss(self.margin,
                self.size_average)(input1, input2, target)


class MultiMarginLoss(Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(MultiMarginLoss, self).__init__()
        if p != 1 and p != 2:
            raise ValueError("only p == 1 and p == 2 supported")
        assert weight is None or weight.dim() == 1
        self.p = p
        self.margin = margin
        self.size_average = size_average
        self.weight = weight

    def forward(self, input, target):
        return self._backend.MultiMarginLoss(self.size_average, self.p,
                self.margin, weight=self.weight)(input, target)


# TODO: L1HingeEmbeddingCriterion
# TODO: MSECriterion weight
# TODO: ClassSimplexCriterion
