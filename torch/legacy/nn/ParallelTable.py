import torch
from torch.legacy import nn

class ParallelTable(nn.Container):

    def __init__(self, ):
        super(ParallelTable, self).__init__()
        self.modules = []
        self.output = []
        self.gradInput = []


    def updateOutput(self, input):
        for i in range(len(self.modules)):
            tmp = self.modules[i].updateOutput(input[i])
            if i in self.output:
                self.output[i] = tmp
            else:
                self.output.append(tmp)

        return self.output

    def updateGradInput(self, input, gradOutput):
        for i, module in enumerate(self.modules):
           self.gradInput[i] = module.updateGradInput(input[i], gradOutput[i])

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        for i, module in ipairs(self.modules):
            module.accGradParameters(input[i], gradOutput[i], scale)

    def accUpdateGradParameters(self, input, gradOutput, lr=1):
        for i, module in ipairs(self.modules):
            module.accUpdateGradParameters(input[i], gradOutput[i], lr)

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = '  |`-> '
        ext = '  |    '
        extlast = '       '
        last = '   ... -> '
        res = torch.type(self)
        res = res + ' {' + line + tab + 'input'
        for i in range(len(self.modules)):
           if i == self.modules-1:
              res = res + line + tab + next + '(' + i + '): ' + toresing(self.modules[i]).gsub(line, line + tab + extlast)
           else:
              res = res + line + tab + next + '(' + i + '): ' + toresing(self.modules[i]).gsub(line, line + tab + ext)


        res = res + line + tab + last + 'output'
        res = res + line + '}'
        return res

