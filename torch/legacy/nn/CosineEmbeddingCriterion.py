import torch
from torch.legacy import nn

class CosineEmbeddingCriterion(nn.Criterion):

    def __init__(self, margin=0, sizeAverage=True):
        super(CosineEmbeddingCriterion, self).__init__()
        self.margin = margin
        self.sizeAverage = sizeAverage
        self.gradInput = [torch.Tensor(), torch.Tensor()]
        self.buffer = None
        self.w1  = None
        self.w22 = None
        self.w  = None
        self.w32 = None
        self._outputs = None
        self._idx = None


    def updateOutput(self, input, y):
        input1, input2 = input[0], input[1]

        # keep backward compatibility
        if not self.buffer:
            self.buffer = input1.new()
            self.w1  = input1.new()
            self.w22 = input1.new()
            self.w  = input1.new()
            self.w32 = input1.new()
            self._outputs = input1.new()

            # comparison operators behave differently from cuda/c implementations
            # TODO: verify name
            if input1.type() == 'torch.cuda.FloatTensor':
                self._idx = input1.new()
            else:
                self._idx = torch.ByteTensor()

        self.buffer.cmul(input1, input2)
        self.w1.sum(self.buffer, 1)

        epsilon = 1e-12
        self.buffer.cmul(input1, input1)
        self.w22.sum(self.buffer, 1).add(epsilon)
        # self._outputs is also used as a temporary buffer
        self._outputs.resizeAs(self.w22).fill(1)
        self.w22.cdiv(self._outputs, self.w22)
        self.w.resizeAs(self.w22).copy(self.w22)

        self.buffer.cmul(input2, input2)
        self.w32.sum(self.buffer, 1).add(epsilon)
        self.w32.cdiv(self._outputs, self.w32)
        self.w.cmul(self.w32)
        self.w.sqrt()

        self._outputs.cmul(self.w1, self.w)
        self._outputs = self._outputs.select(1, 0)

        torch.eq(self._idx, y, -1)
        self._outputs[self._idx] = self._outputs[self._idx].add(-self.margin).cmax(0)
        torch.eq(self._idx, y, 1)
        self._outputs[self._idx] = self._outputs[self._idx].mul(-1).add(1)

        self.output = self._outputs.sum()

        if self.sizeAverage:
           self.output = self.output / y.size(0)

        return self.output


    def updateGradInput(self, input, y):
        v1  = input[0]
        v2  = input[1]

        gw1 = self.gradInput[0]
        gw2 = self.gradInput[1]
        gw1.resizeAs(v1).copy(v2)
        gw2.resizeAs(v1).copy(v1)

        self.buffer.cmul(self.w1, self.w22)
        gw1.addcmul(-1, self.buffer.expandAs(v1), v1)
        gw1.cmul(self.w.expandAs(v1))

        self.buffer.cmul(self.w1, self.w32)
        gw2.addcmul(-1, self.buffer.expandAs(v1), v2)
        gw2.cmul(self.w.expandAs(v1))

        # self._idx = self._outputs <= 0
        torch.le(self._idx, self._outputs, 0)
        self._idx = self._idx.view(-1, 1).expand(gw1.size())
        gw1[self._idx] = 0
        gw2[self._idx] = 0

        torch.eq(self._idx, y,1)
        self._idx = self._idx.view(-1, 1).expand(gw2.size())
        gw1[self._idx] = gw1[self._idx].mul(-1)
        gw2[self._idx] = gw2[self._idx].mul(-1)

        if self.sizeAverage:
           gw1.div(y.size(0))
           gw2.div(y.size(0))

        return self.gradInput

    def type(self, type=None, tensorCache=None):
        if not type:
           return self._type

        self._idx = nil
        super(CosineEmbeddingCriterion, self).type(self, type, tensorCache)
        # comparison operators behave differently from cuda/c implementations
        # TODO: verify name
        if type == 'torch.cuda.FloatTensor':
           self._idx = torch.cuda.FloatTensor()
        else:
           self._idx = torch.ByteTensor()

        return self

