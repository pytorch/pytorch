import torch
from torch.legacy import nn

class MixtureTable(nn.Module):

    def __init__(self, dim=1):
        super(MixtureTable, self).__init__()
        self.dim = dim
        self.size = torch.LongStorage()
        self.batchSize = 0
        self.size2 = torch.LongStorage()
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
                self.size.resize(expertInput.dim()+1).fill(1)
                if self.dimG > 0:
                    self.size[0] = gaterInput.size(0)

                self.size[self.dim] = gaterInput.size(self.dimG)
                self.output.resizeAs(expertInput)
                self.backwardSetup = False
                self.batchSize = batchSize

            self._gaterView.view(gaterInput, self.size)
            self.output.zero()
            # multiply accumulate gater outputs by their commensurate expert
            for i, expertInput in enumerate(expertInputs):
                gate = self._gaterView.select(self.dim, i).expandAs(expertInput)
                self.output.addcmul(expertInput, gate)
        else:
            if self.batchSize != batchSize:
                self.size.resize(expertInputs.dim()).fill(1)
                if self.dimG > 0:
                    self.size[0] = gaterInput.size(0)

                self.size[self.dim] = gaterInput.size(self.dimG)
                self.output.resizeAs(expertInputs.select(self.dim, 0))
                self.batchSize = batchSize
                self.backwardSetup = False

            self._gaterView.view(gaterInput, self.size)
            self._expert.cmul(self._gaterView.expandAs(expertInputs), expertInputs)
            self.output.sum(self._expert, self.dim)
            self.output.resizeAs(expertInputs.select(self.dim, 0))

        return self.output


    def updateGradInput(self, input, gradOutput):
        gaterInput, expertInputs = input
        nn.utils.recursiveResizeAs(self.gradInput, input)
        gaterGradInput, expertGradInputs = self.gradInput

        # buffers
        self._sum = self._sum or input[0].new()
        self._expertView2 = self._expertView2 or input[0].new()
        self._expert2 = self._expert2 or input[0].new()

        if self.table:
            if not self.backwardSetup:
                for i, expertInput in enumerate(expertInputs):
                    expertGradInput = expertGradInputs[i] or expertInput.clone()
                    expertGradInput.resizeAs(expertInput)
                    expertGradInputs[i] = expertGradInput

                gaterGradInput.resizeAs(gaterInput)
                self.backwardSetup = True


            # like CMulTable, but with broadcasting
            for i, expertGradInput in enumerate(expertGradInputs):
                # gater updateGradInput
                self._expert.cmul(gradOutput, expertInputs[i])
                if self.dimG == 0:
                    self._expertView.view(self._expert, -1)
                else:
                    self._expertView.view(self._expert, gradOutput.size(0), -1)

                self._sum.sum(self._expertView, self.dimG)
                if self.dimG == 0:
                    gaterGradInput[i] = self._sum.select(self.dimG, 0)
                else:
                    gaterGradInput.select(self.dimG, i).copy(self._sum.select(self.dimG, 0))

                # expert updateGradInput
                gate = self._gaterView.select(self.dim, i).expandAs(expertGradInput)
                expertGradInput.cmul(gate, gradOutput)
        else:
            if not self.backwardSetup:
                self.size2.resize(expertInputs.dim())
                self.size2.copy(expertInputs.size())
                self.size2[self.dim] = 1
                gaterGradInput.resizeAs(gaterInput)
                self.backwardSetup = True

            # gater updateGradInput
            self._expertView.view(gradOutput, self.size2)
            gradOutput = self._expertView.expandAs(expertInputs)
            self._expert.cmul(gradOutput, expertInputs)
            expert = self._expert.transpose(self.dim, self.dimG)
            if not expert.isContiguous():
                self._expert2.resizeAs(expert)
                self._expert2.copy(expert)
                expert = self._expert2
            if self.dimG == 0:
                self._expertView2.view(expert, gaterInput.size(0), -1)
            else:
                self._expertView2.view(expert, gaterInput.size(0), gaterInput.size(1), -1)

            gaterGradInput.sum(self._expertView2, self.dimG+1)
            gaterGradInput.resizeAs(gaterInput)

            # expert updateGradInput
            expertGradInputs.cmul(self._gaterView.expandAs(expertInputs), gradOutput)

        return self.gradInput


    def type(self, type, tensorCache):
        self._gaterView = None
        self._expert = None
        self._expertView = None
        self._sum = None
        self._expert2 = None
        self._expertView2 = None
        return super(MixtureTable, self).type(type, tensorCache)


    def clearState(self, ):
        nn.utils.clear(self, [
          '_gaterView',
          '_expert',
          '_expertView',
          '_sum',
          '_expert2',
          '_expertView2',
        ])
        return super(MixtureTable, self).clearState()

