import torch
from torch.legacy import nn

class MultiCriterion(nn.Module):

    def __init__(self, ):
        super(MultiCriterion, self).__init__()
        self.criterions = []
        self.weights = torch.DoubleStorage()

    def add(self, criterion, weight=1):
        self.criterions.append(criterion)
        self.weights.resize(len(self.criterions), True)
        self.weights[len(self.criterions)-1] = weight
        return self

    def updateOutput(self, input, target):
        self.output = 0
        for i in range(len(self.criterions)):
            self.output = self.output + self.weights[i] * self.criterions[i].updateOutput(input, target)

        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
        nn.utils.recursiveFill(self.gradInput, 0)
        for i in range(len(self.criterions)):
           nn.utils.recursiveAdd(self.gradInput, self.weights[i], self.criterions[i].updateGradInput(input, target))

        return self.gradInput

    def type(self, type):
        for i, criterion in ipairs(self.criterions):
           criterion.type(type)

        return super(MultiCriterion, self).type(type)

