import torch
from .Criterion import Criterion
from .utils import recursiveResizeAs, recursiveFill, recursiveAdd


class ParallelCriterion(Criterion):

    def __init__(self, repeatTarget=False):
        super(ParallelCriterion, self).__init__()
        self.criterions = []
        self.weights = []
        self.gradInput = []
        self.repeatTarget = repeatTarget

    def add(self, criterion, weight=1):
        self.criterions.append(criterion)
        self.weights.append(weight)
        return self

    def updateOutput(self, input, target):
        self.output = 0
        for i, criterion in enumerate(self.criterions):
            current_target = target if self.repeatTarget else target[i]
            self.output += self.weights[i] * criterion.updateOutput(input[i], current_target)

        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = recursiveResizeAs(self.gradInput, input)[0]
        recursiveFill(self.gradInput, 0)
        for i, criterion in enumerate(self.criterions):
            current_target = target if self.repeatTarget else target[i]
            recursiveAdd(self.gradInput[i], self.weights[i], criterion.updateGradInput(input[i], current_target))

        return self.gradInput

    def type(self, type=None, tensorCache=None):
        self.gradInput = []
        return super(ParallelCriterion, self).type(type, tensorCache)
