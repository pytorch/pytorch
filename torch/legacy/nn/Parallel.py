import torch
from .Container import Container


class Parallel(Container):

    def __init__(self, inputDimension, outputDimension):
        super(Parallel, self).__init__()
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        self.totalOutputSize = None

    def updateOutput(self, input):
        nModule = input.size(self.inputDimension)
        outputs = []

        for i in range(nModule):
            currentInput = input.select(self.inputDimension, i)
            currentOutput = self.modules[i].updateOutput(currentInput)
            outputs.append(currentOutput)
            outputSize = currentOutput.size(self.outputDimension)

            if i == 0:
                totalOutputSize = list(currentOutput.size())
            else:
                totalOutputSize[self.outputDimension] += outputSize

        self.totalOutputSize = torch.Size(totalOutputSize)
        self.output.resize_(self.totalOutputSize)

        offset = 0
        for i in range(nModule):
            currentOutput = outputs[i]
            outputSize = currentOutput.size(self.outputDimension)
            self.output.narrow(self.outputDimension, offset, outputSize).copy_(currentOutput)
            offset = offset + currentOutput.size(self.outputDimension)

        return self.output

    def updateGradInput(self, input, gradOutput):
        nModule = input.size(self.inputDimension)
        self.gradInput.resize_as_(input)

        offset = 0
        for i in range(nModule):
            module = self.modules[i]
            currentInput = input.select(self.inputDimension, i)
            currentOutput = module.output
            outputSize = currentOutput.size(self.outputDimension)
            currentGradOutput = gradOutput.narrow(self.outputDimension, offset, outputSize)

            currentGradInput = module.updateGradInput(currentInput, currentGradOutput)

            self.gradInput.select(self.inputDimension, i).copy_(currentGradInput)
            offset = offset + outputSize

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        nModule = input.size(self.inputDimension)

        offset = 0
        for i in range(nModule):
            module = self.modules[i]
            currentOutput = module.output
            outputSize = currentOutput.size(self.outputDimension)

            module.accGradParameters(
                input.select(self.inputDimension, i),
                gradOutput.narrow(self.outputDimension, offset, outputSize),
                scale)
            offset += outputSize

    def accUpdateGradParameters(self, input, gradOutput, lr):
        nModule = input.size(self.inputDimension)

        offset = 0
        for i in range(nModule):
            module = self.modules[i]
            currentOutput = module.output
            module.accupdateGradParameters(
                input.select(self.inputDimension, i),
                gradOutput.narrow(self.outputDimension, offset, currentOutput.size(self.outputDimension)),
                lr)
            offset = offset + currentOutput.size(self.outputDimension)

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = '  |`-> '
        ext = '  |    '
        extlast = '       '
        last = '   ... -> '
        res = torch.typename(self)
        res += ' {' + line + tab + 'input'
        for i in range(len(self.modules)):
            if i == len(self.modules) - 1:
                res += line + tab + next + '(' + str(i) + '): ' + \
                    str(self.modules[i]).replace(line, line + tab + extlast)
            else:
                res += line + tab + next + '(' + str(i) + '): ' + str(self.modules[i]).replace(line, line + tab + ext)

        res += line + tab + last + 'output'
        res += line + '}'
        return res
