import torch
from .Criterion import Criterion


class L1HingeEmbeddingCriterion(Criterion):

    def __init__(self, margin=1):
        super(L1HingeEmbeddingCriterion, self).__init__()
        self.margin = margin
        self.gradInput = [torch.Tensor(), torch.Tensor()]

    def updateOutput(self, input, y):
        self.output = input[0].dist(input[1], 1)
        if y == -1:
            self.output = max(0, self.margin - self.output)

        return self.output

    def _mathsign(t):
        return 1 if x > 0 else -1

    def updateGradInput(self, input, y):
        self.gradInput[0].resize_as_(input[0])
        self.gradInput[1].resize_as_(input[1])
        self.gradInput[0].copy_(input[0])
        self.gradInput[0].add_(-1, input[1])
        dist = self.gradInput[0].norm(1)
        self.gradInput[0].sign_()
        if y == -1:  # just to avoid a mul by 1
            if dist > self.margin:
                self.gradInput[0].zero_()
            else:
                self.gradInput[0].mul_(-1)

        self.gradInput[1].zero_().add_(-1, self.gradInput[0])
        return self.gradInput
