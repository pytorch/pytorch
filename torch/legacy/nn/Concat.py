import torch
from torch.legacy import nn

class Concat(nn.Container):

    def __init__(self, dimension):
        super(Concat, self).__init__()
        self.size = torch.LongStorage()
        self.dimension = dimension

    def updateOutput(self, input):
        outs = []
        for i in range(len(self.modules)):
            currentOutput = self.modules[i].updateOutput(input)
            outs.append(currentOutput)
            if i == 0:
                self.size.resize_(currentOutput.dim()).copy_(currentOutput.size())
            else:
                self.size[self.dimension] = self.size[self.dimension] + currentOutput.size(self.dimension)

        self.output.resize_(self.size)

        offset = 0
        for i, module in enumerate(self.modules):
           currentOutput = outs[i]
           self.output.narrow(self.dimension, offset, currentOutput.size(self.dimension)).copy_(currentOutput)
           offset = offset + currentOutput.size(self.dimension)

        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resizeAs_(input)

        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = module.output
            currentGradInput = module.updateGradInput(input, gradOutput.narrow(self.dimension, offset, currentOutput.size(self.dimension)))

            if currentGradInput: # if the module does not produce a gradInput (for example first layer),: ignore it and move on.
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
        self.gradInput.resizeAs_(input)
        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = module.output
            currentGradInput = module.backward(input, gradOutput.narrow(self.dimension, offset, currentOutput.size(self.dimension)), scale)
            if currentGradInput: # if the module.es not produce a gradInput (for example first layer),: ignore it and move on.
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
            if i == len(self.modules)-1:
                res += line + tab + next + '(' + i + '): ' + str(self.modules[i]).replace(line, line + tab + extlast)
            else:
                res += line + tab + next + '(' + i + '): ' + str(self.modules[i]).replace(line, line + tab + ext)

        res += line + tab + last + 'output'
        res += line + '}'
        return res


