import torch
from torch.legacy import nn

class MV(nn.Module):
    """Module to perform matrix vector multiplication on two minibatch inputs,
       producing a minibatch.
    """

    def __init__(self, trans=False):
        super(MV, self).__init__()

        self.trans = trans
        self.gradInput = [torch.Tensor(), torch.Tensor()]

    def updateOutput(self, input):
        M, v = input
        assert M.nDimension() == 2 or M.nDimension() == 3

        if M.nDimension() == 2:
            assert v.nDimension() == 1
            if self.trans:
                M = M.transpose(0, 1)
            self.output.resize(M.size(0))
            self.output.mv(M, v)
        else:
            assert v.nDimension() == 2
            if self.trans:
                M = M.transpose(1, 2)
            self.output.resize(M.size(0), M.size(1), 1)
            self.output.bmm(M, v.view(v.size(0), v.size(1), 1)).resize(M.size(0), M.size(1))

        return self.output

    def updateGradInput(self, input, gradOutput):
        M, v = input
        self.gradInput[0].resizeAs(M)
        self.gradInput[1].resizeAs(v)

        assert gradOutput.nDimension() == 1 or gradOutput.nDimension() == 2

        if gradOutput.nDimension() == 2:
            assert M.nDimension() == 3
            assert v.nDimension() == 2
            bdim = M.size(0)
            odim = M.size(1)
            idim = M.size(2)

            if self.trans:
                self.gradInput[0].bmm(v.view(bdim, odim, 1), gradOutput.view(bdim, 1, idim))
                self.gradInput[1].view(bdim, odim, 1).bmm(M, gradOutput.view(bdim, idim, 1))
            else:
                self.gradInput[0].bmm(gradOutput.view(bdim, odim, 1), v.view(bdim, 1, idim))
                self.gradInput[1].view(bdim, idim, 1).bmm(M.transpose(1, 2), gradOutput.view(bdim, odim, 1))
        else:
            assert M.nDimension() == 2
            assert v.nDimension() == 1

            if self.trans:
                self.gradInput[0].ger(v, gradOutput)
                self.gradInput[1] = M * gradOutput
            else:
                self.gradInput[0].ger(gradOutput, v)
                self.gradInput[1] = M.t() * gradOutput

        return self.gradInput

