import torch
from .Module import Module
from .utils import clear, recursiveResizeAs


class MixtureTable(Module):

    def __init__(self, dim=1):
        super(MixtureTable, self).__init__()
        self.dim = dim
        self.size = torch.Size()
        self.size2 = torch.Size()
        self.batchSize = 0
        self.backwardSetup = False
        self.gradInput = []

        self._gaterView = None
        self._expert = None
        self._expertView = None
        self._sum = None
        self._expertView2 = None
        self._expert2 = None
        self.table = False

    def updateOutput(self, input):
        gaterInput, expertInputs = input

        # buffers
        self._gaterView = self._gaterView or input[0].new()
        self._expert = self._expert or input[0].new()
        self._expertView = self._expertView or input[0].new()

        self.dimG = 1
        batchSize = gaterInput.size(0)

        if self.table or isinstance(expertInputs, list):
            self.table = True
            if gaterInput.size(self.dimG) != len(expertInputs):
                raise RuntimeError("Should be one gater output per expert")

            expertInput = expertInputs[0]
            if self.batchSize != batchSize:
                size = [1] * (expertInput.dim()+1)
                if self.dimG > 0:
                    size[0] = gaterInput.size(0)
                size[self.dim] = gaterInput.size(self.dimG)
                self.size = torch.Size(size)
                self.output.resize_as_(expertInput)
                self.backwardSetup = False
                self.batchSize = batchSize

            self._gaterView = gaterInput.view(self.size)
            self.output.zero_()
            # multiply accumulate gater outputs by their commensurate expert
            for i, expertInput in enumerate(expertInputs):
                gate = self._gaterView.select(self.dim, i).expand_as(expertInput)
                self.output.addcmul_(expertInput, gate)
        else:
            if self.batchSize != batchSize:
                size = [1] * expertInputs.dim()
                if self.dimG > 0:
                    size[0] = gaterInput.size(0)
                size[self.dim] = gaterInput.size(self.dimG)
                self.size = torch.Size(size)
                self.output.resize_as_(expertInputs.select(self.dim, 0))
                self.batchSize = batchSize
                self.backwardSetup = False

            self._gaterView = gaterInput.view(self.size)
            torch.mul(self._expert, self._gaterView.expand_as(expertInputs), expertInputs)
            torch.sum(self.output, self._expert, self.dim)
            self.output.resize_as_(expertInputs.select(self.dim, 0))

        return self.output

    def updateGradInput(self, input, gradOutput):
        gaterInput, expertInputs = input
        recursiveResizeAs(self.gradInput, input)
        gaterGradInput, expertGradInputs = self.gradInput

        # buffers
        self._sum = self._sum or input[0].new()
        self._expertView2 = self._expertView2 or input[0].new()
        self._expert2 = self._expert2 or input[0].new()

        if self.table:
            if not self.backwardSetup:
                for i, expertInput in enumerate(expertInputs):
                    expertGradInput = expertGradInputs[i] or expertInput.clone()
                    expertGradInput.resize_as_(expertInput)
                    expertGradInputs[i] = expertGradInput

                gaterGradInput.resize_as_(gaterInput)
                self.backwardSetup = True


            # like CMulTable, but with broadcasting
            for i, expertGradInput in enumerate(expertGradInputs):
                # gater updateGradInput
                torch.mul(self._expert, gradOutput, expertInputs[i])
                if self.dimG == 0:
                    self._expertView = self._expert.view(-1)
                else:
                    self._expertView = self._expert.view(gradOutput.size(0), -1)

                torch.sum(self._sum, self._expertView, self.dimG)
                if self.dimG == 0:
                    gaterGradInput[i] = self._sum.select(self.dimG, 0)
                else:
                    gaterGradInput.select(self.dimG, i).copy_(self._sum.select(self.dimG, 0))

                # expert updateGradInput
                gate = self._gaterView.select(self.dim, i).expand_as(expertGradInput)
                expertGradInput.mul_(gate, gradOutput)
        else:
            if not self.backwardSetup:
                size2 = list(expertInputs.size())
                size2[self.dim] = 1
                self.size2 = torch.Size(size2)
                gaterGradInput.resize_as_(gaterInput)
                self.backwardSetup = True

            # gater updateGradInput
            self._expertView = gradOutput.view(torch.Size(self.size2))
            gradOutput = self._expertView.expand_as(expertInputs)
            torch.mul(self._expert, gradOutput, expertInputs)
            expert = self._expert.transpose(self.dim, self.dimG)
            if not expert.is_contiguous():
                self._expert2.resize_as_(expert)
                self._expert2.copy_(expert)
                expert = self._expert2
            if self.dimG == 0:
                self._expertView2 = expert.view(gaterInput.size(0), -1)
            else:
                self._expertView2 = expert.view(gaterInput.size(0), gaterInput.size(1), -1)

            torch.sum(gaterGradInput, self._expertView2, self.dimG+1)
            gaterGradInput.resize_as_(gaterInput)

            # expert updateGradInput
            torch.mul(expertGradInputs, self._gaterView.expand_as(expertInputs), gradOutput)

        return self.gradInput

    def type(self, type, tensorCache=None):
        self._gaterView = None
        self._expert = None
        self._expertView = None
        self._sum = None
        self._expert2 = None
        self._expertView2 = None
        return super(MixtureTable, self).type(type, tensorCache)

    def clearState(self, ):
        clear(self, [
          '_gaterView',
          '_expert',
          '_expertView',
          '_sum',
          '_expert2',
          '_expertView2',
        ])
        return super(MixtureTable, self).clearState()
