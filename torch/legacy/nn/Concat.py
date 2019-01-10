import torch
from .Container import Container


class Concat(Container):

    def __init__(self, dimension):
        super(Concat, self).__init__()
        self.outputSize = torch.Size()
        self.dimension = dimension

    def updateOutput(self, input):
        outs = []
        for i in range(len(self.modules)):
            currentOutput = self.modules[i].updateOutput(input)
            outs.append(currentOutput)
            if i == 0:
                size = list(currentOutput.size())
            else:
                size[self.dimension] += currentOutput.size(self.dimension)
        self.outputSize = torch.Size(size)
        self.output.resize_(self.outputSize)

        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = outs[i]
            self.output.narrow(self.dimension, offset, currentOutput.size(self.dimension)).copy_(currentOutput)
            offset = offset + currentOutput.size(self.dimension)

        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_as_(input)

        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = module.output
            currentGradInput = module.updateGradInput(input, gradOutput.narrow(
                self.dimension, offset, currentOutput.size(self.dimension)))

            # if the module does not produce a gradInput (for example first layer),: ignore it and move on.
            if currentGradInput:
                if i == 0:
                    self.gradInput.copy_(currentGradInput)
                else:
                    self.gradInput.add_(currentGradInput)

            offset = offset + currentOutput.size(self.dimension)

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = module.output
            module.accGradParameters(
                input,
                gradOutput.narrow(self.dimension, offset, currentOutput.size(self.dimension)),
                scale)
            offset = offset + currentOutput.size(self.dimension)

    def backward(self, input, gradOutput, scale=1):
        self.gradInput.resize_as_(input)
        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = module.output
            currentGradInput = module.backward(input, gradOutput.narrow(
                self.dimension, offset, currentOutput.size(self.dimension)), scale)
            # if the module.es not produce a gradInput (for example first layer),: ignore it and move on.
            if currentGradInput is not None:
                if i == 0:
                    self.gradInput.copy_(currentGradInput)
                else:
                    self.gradInput.add_(currentGradInput)
            offset = offset + currentOutput.size(self.dimension)

        return self.gradInput

    def accUpdateGradParameters(self, input, gradOutput, lr):
        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = module.output
            module.accUpdateGradParameters(
                input,
                gradOutput.narrow(self.dimension, offset, currentOutput.size(self.dimension)),
                lr)
            offset = offset + currentOutput.size(self.dimension)

    def __tostring__(self):
        tab = '  '
        line = '\n'
        next = '  |`-> '
        ext = '  |    '
        extlast = '       '
        last = '   +. -> '
        res = torch.type(self)
        res += ' {' + line + tab + 'input'
        for i in range(len(self.modules)):
            if i == len(self.modules) - 1:
                res += line + tab + next + '(' + i + '): ' + str(self.modules[i]).replace(line, line + tab + extlast)
            else:
                res += line + tab + next + '(' + i + '): ' + str(self.modules[i]).replace(line, line + tab + ext)

        res += line + tab + last + 'output'
        res += line + '}'
        return res
