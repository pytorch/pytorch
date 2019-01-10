import torch
from .Criterion import Criterion
from .utils import recursiveResizeAs, recursiveFill, recursiveAdd


class MultiCriterion(Criterion):

    def __init__(self, ):
        super(MultiCriterion, self).__init__()
        self.criterions = []
        self.weights = torch.DoubleStorage()

    def add(self, criterion, weight=1):
        self.criterions.append(criterion)
        new_weights = torch.DoubleStorage(len(self.criterions))
        for i, v in enumerate(self.weights):
            new_weights[i] = v
        new_weights[len(self.criterions) - 1] = weight
        self.weights = new_weights
        return self

    def updateOutput(self, input, target):
        self.output = 0
        for i in range(len(self.criterions)):
            self.output = self.output + self.weights[i] * self.criterions[i].updateOutput(input, target)

        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = recursiveResizeAs(self.gradInput, input)[0]
        recursiveFill(self.gradInput, 0)
        for i in range(len(self.criterions)):
            recursiveAdd(self.gradInput, self.weights[i], self.criterions[i].updateGradInput(input, target))

        return self.gradInput

    def type(self, type):
        for criterion in self.criterions:
            criterion.type(type)

        return super(MultiCriterion, self).type(type)
