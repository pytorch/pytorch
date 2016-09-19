import torch
from .Module import Module

class MV(Module):
    """Module to perform matrix vector multiplication on two minibatch inputs,
       producing a minibatch.
    """

    def __init__(self, trans=False):
        super(MV, self).__init__()

        self.trans = trans
        self.gradInput = [torch.Tensor(), torch.Tensor()]

    def updateOutput(self, input):
        M, v = input
        assert M.ndimension() == 2 or M.ndimension() == 3

        if M.ndimension() == 2:
            assert v.ndimension() == 1
            if self.trans:
                M = M.transpose(0, 1)
            self.output.resize_(M.size(0))
            torch.mv(self.output, M, v)
        else:
            assert v.ndimension() == 2
            if self.trans:
                M = M.transpose(1, 2)
            self.output.resize_(M.size(0), M.size(1), 1)
            torch.bmm(self.output, M, v.view(v.size(0), v.size(1), 1)).resize_(M.size(0), M.size(1))

        return self.output

    def updateGradInput(self, input, gradOutput):
        M, v = input
        self.gradInput[0].resize_as_(M)
        self.gradInput[1].resize_as_(v)

        assert gradOutput.ndimension() == 1 or gradOutput.ndimension() == 2

        if gradOutput.ndimension() == 2:
            assert M.ndimension() == 3
            assert v.ndimension() == 2
            bdim = M.size(0)
            odim = M.size(1)
            idim = M.size(2)

            if self.trans:
                torch.bmm(self.gradInput[0], v.view(bdim, odim, 1), gradOutput.view(bdim, 1, idim))
                torch.bmm(self.gradInput[1].view(bdim, odim, 1), M, gradOutput.view(bdim, idim, 1))
            else:
                torch.bmm(self.gradInput[0], gradOutput.view(bdim, odim, 1), v.view(bdim, 1, idim))
                torch.bmm(self.gradInput[1].view(bdim, idim, 1), M.transpose(1, 2), gradOutput.view(bdim, odim, 1))
        else:
            assert M.ndimension() == 2
            assert v.ndimension() == 1

            if self.trans:
                torch.ger(self.gradInput[0], v, gradOutput)
                self.gradInput[1] = M * gradOutput
            else:
                torch.ger(self.gradInput[0], gradOutput, v)
                self.gradInput[1] = M.t() * gradOutput

        return self.gradInput

