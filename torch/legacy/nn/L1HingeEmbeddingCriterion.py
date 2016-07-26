import torch
from torch.legacy import nn

class L1HingeEmbeddingCriterion(nn.Criterion):

    def __init__(self, margin=1):
        super(L1HingeEmbeddingCriterion, self).__init__()
        self.margin = margin
        self.gradInput = [torch.Tensor(), torch.Tensor()]

    def updateOutput(self, input, y):
        self.output = input[0].dist(input[1], 1);
        if y == -1:
            self.output = max(0, self.margin - self.output);

        return self.output

    def _mathsign(t):
        return 1 if x > 0 else -1

    def updateGradInput(self, input, y):
        self.gradInput[0].resizeAs(input[0])
        self.gradInput[1].resizeAs(input[1])
        self.gradInput[0].copy(input[0])
        self.gradInput[0].add(-1, input[1])
        dist = self.gradInput[0].norm(1);
        self.gradInput[0].sign()
        if y == -1:  # just to avoid a mul by 1
            if dist > self.margin:
                self.gradInput[0].zero()
            else:
                self.gradInput[0].mul(-1)

        self.gradInput[1].zero().add(-1, self.gradInput[0])
        return self.gradInput

