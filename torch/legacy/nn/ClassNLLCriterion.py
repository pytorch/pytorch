import torch
from torch.legacy import nn

class ClassNLLCriterion(nn.Criterion):
    def __init__(self, weights=None, sizeAverage=True):
        super(ClassNLLCriterion, self).__init__()
        self.sizeAverage = sizeAverage

        if weights is not None:
            assert weights.dim() == 1
        self.weights = weights

        self.output_tensor = torch.zeros(1)
        self.total_weight_tensor = torch.ones(1)
        self.target = torch.zeros(1).long()

    def updateOutput(self, input, target):
        if target.type() == 'torch.cuda.CudaTensor':
            self.target = target
        else:
            self.target = target.long()


        self._backend.ClassNLLCriterion_updateOutput(
            self._backend.library_state,
            input,
            self.target,
            self.output_tensor,
            self.sizeAverage,
            self.weights,
            self.total_weight_tensor
        )
        self.output = self.output_tensor[0]
        return self.output


    def updateGradInput(self, input, target):
        if target.type() == 'torch.cuda.FloatTensor':
            self.target = target
        else:
            self.target = target.long()

        self.gradInput.resizeAs(input).zero()

        self._backend.ClassNLLCriterion_updateGradInput(
            self._backend.library_state,
            input,
            self.target,
            self.gradInput,
            self.sizeAverage,
            self.weights,
            self.total_weight_tensor
        )

        return self.gradInput

