import torch
from .Container import Container


class ParallelTable(Container):

    def __init__(self, ):
        super(ParallelTable, self).__init__()
        self.modules = []
        self.output = []
        self.gradInput = []

    def updateOutput(self, input):
        for i in range(len(self.modules)):
            tmp = self.modules[i].updateOutput(input[i])
            if len(self.output) <= i:
                self.output.append(tmp)
            else:
                self.output[i] = tmp

        return self.output

    def updateGradInput(self, input, gradOutput):
        for i, module in enumerate(self.modules):
            tmp = module.updateGradInput(input[i], gradOutput[i])
            if len(self.gradInput) <= i:
                self.gradInput.append(tmp)
            else:
                self.gradInput[i] = tmp

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        for i, module in enumerate(self.modules):
            module.accGradParameters(input[i], gradOutput[i], scale)

    def accUpdateGradParameters(self, input, gradOutput, lr=1):
        for i, module in enumerate(self.modules):
            module.accUpdateGradParameters(input[i], gradOutput[i], lr)

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = '  |`-> '
        ext = '  |    '
        extlast = '       '
        last = '   ... -> '
        res = torch.typename(self)
        res = res + ' {' + line + tab + 'input'
        for i in range(len(self.modules)):
            if i == len(self.modules) - 1:
                res = res + line + tab + next + '(' + str(i) + '): ' + \
                    str(self.modules[i]).replace(line, line + tab + extlast)
            else:
                res = res + line + tab + next + '(' + str(i) + '): ' + \
                    str(self.modules[i]).replace(line, line + tab + ext)

        res = res + line + tab + last + 'output'
        res = res + line + '}'
        return res
