import torch
from .Container import Container


class Sequential(Container):

    def __len__(self):
        return len(self.modules)

    def add(self, module):
        if len(self.modules) == 0:
            self.gradInput = module.gradInput

        self.modules.append(module)
        self.output = module.output
        return self

    def insert(self, module, index):
        self.modules.insert(module, index)
        self.output = self.modules[-1].output
        self.gradInput = self.modules[0].gradInput

    def remove(self, index=-1):
        del self.modules[index]

        if len(self.modules) > 0:
            self.output = self.modules[-1].output
            self.gradInput = self.modules[0].gradInput
        else:
            self.output = torch.Tensor()
            self.gradInput = torch.Tensor()

    def updateOutput(self, input):
        currentOutput = input
        for i, module in enumerate(self.modules):
            currentOutput = module.updateOutput(currentOutput)
        self.output = currentOutput
        return self.output

    def _iter_with_prev(self):
        return zip(self.modules[-2::-1], self.modules[-1:0:-1])

    def updateGradInput(self, input, gradOutput):
        currentGradOutput = gradOutput
        for prev, current in self._iter_with_prev():
            currentGradOutput = current.updateGradInput(prev.output, currentGradOutput)
        self.gradInput = self.modules[0].updateGradInput(input, currentGradOutput)
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        currentGradOutput = gradOutput
        for prev, current in self._iter_with_prev():
            current.accGradParameters(prev.output, currentGradOutput, scale)
            currentGradOutput = current.gradInput
        self.modules[0].accGradParameters(input, currentGradOutput, scale)

    def backward(self, input, gradOutput, scale=1):
        currentGradOutput = gradOutput
        for prev, current in self._iter_with_prev():
            currentGradOutput = current.backward(prev.output, currentGradOutput, scale)
            # currentModule.gradInput = currentGradOutput
        self.gradInput = self.modules[0].backward(input, currentGradOutput, scale)
        return self.gradInput

    def accUpdateGradParameters(self, input, gradOutput, lr):
        currentGradOutput = gradOutput
        for prev, current in self._iter_with_prev():
            current.accUpdateGradParameters(prev.output, currentGradOutput, lr)
            currentGradOutput = current.gradInput
        self.modules[0].accUpdateGradParameters(input, currentGradOutput, lr)

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = ' -> '
        res = 'nn.Sequential'
        res = res + ' {' + line + tab + '[input'
        for i in range(len(self.modules)):
            res = res + next + '(' + str(i) + ')'

        res = res + next + 'output]'
        for i in range(len(self.modules)):
            res = res + line + tab + '(' + str(i) + '): ' + str(self.modules[i]).replace(line, line + tab)

        res = res + line + '}'
        return res
