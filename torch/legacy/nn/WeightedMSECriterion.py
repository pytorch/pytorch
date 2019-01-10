import torch
from .Criterion import Criterion


class WeightedMSECriterion(Criterion):

    def __init__(self, weight, sizeAverage=True):
        super(WeightedMSECriterion, self).__init__()
        self.weight = weight.clone()
        self.buffer = None
        self.output_tensor = None
        self.sizeAverage = sizeAverage

    def updateOutput(self, input, target):
        if self.buffer is None:
            self.buffer = input.new()
        self.buffer.resize_as_(input).copy_(target)
        if input.dim() - 1 == self.weight.dim():
            for i in range(input.size(0)):
                self.buffer[i].mul_(self.weight)
        else:
            self.buffer.mul_(self.weight)

        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        self._backend.MSECriterion_updateOutput(
            self._backend.library_state,
            input,
            self.buffer,
            self.output_tensor,
            self.sizeAverage,
            True,  # reduce
        )
        self.output = self.output_tensor[0].item()
        return self.output

    def updateGradInput(self, input, target):
        self.buffer.resize_as_(input).copy_(target)
        if input.dim() - 1 == self.weight.dim():
            for i in range(input.size(0)):
                self.buffer[i].mul_(self.weight)
        else:
            self.buffer.mul_(self.weight)

        implicit_gradOutput = torch.Tensor([1]).type(input.type())

        self._backend.MSECriterion_updateGradInput(
            self._backend.library_state,
            input,
            self.buffer,
            implicit_gradOutput,
            self.gradInput,
            self.sizeAverage,
            True,  # reduce
        )
        return self.gradInput
