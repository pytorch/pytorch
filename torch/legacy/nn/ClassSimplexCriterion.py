import math
import torch
from .MSECriterion import MSECriterion

"""
         This file implements a criterion for multi-class classification.
         It learns an embedding per class, where each class' embedding
         is a point on an (N-1)-dimensional simplex, where N is
         the number of classes.
         For example usage of this class, look at.c/criterion.md

         Reference: http.//arxiv.org/abs/1506.08230
"""


class ClassSimplexCriterion(MSECriterion):

    def __init__(self, nClasses):
        super(ClassSimplexCriterion, self).__init__()
        self.nClasses = nClasses

        # embedding the simplex in a space of dimension strictly greater than
        # the minimum possible (nClasses-1) is critical for effective training.
        simp = self._regsplex(nClasses - 1)
        self.simplex = torch.cat((simp, torch.zeros(simp.size(0), nClasses - simp.size(1))), 1)
        self._target = torch.Tensor(nClasses)

        self.output_tensor = None

    def _regsplex(self, n):
        """
        regsplex returns the coordinates of the vertices of a
        regular simplex centered at the origin.
        The Euclidean norms of the vectors specifying the vertices are
        all equal to 1. The input n is the dimension of the vectors;
        the simplex has n+1 vertices.

        input:
        n # dimension of the vectors specifying the vertices of the simplex

        output:
        a # tensor dimensioned (n+1, n) whose rows are
             vectors specifying the vertices

        reference:
        http.//en.wikipedia.org/wiki/Simplex#Cartesian_coordinates_for_regular_n-dimensional_simplex_in_Rn
        """
        a = torch.zeros(n + 1, n)

        for k in range(n):
            # determine the last nonzero entry in the vector for the k-th vertex
            if k == 0:
                a[k][k] = 1
            else:
                a[k][k] = math.sqrt(1 - a[k:k + 1, 0:k + 1].norm() ** 2)

            # fill_ the k-th coordinates for the vectors of the remaining vertices
            c = (a[k][k] ** 2 - 1 - 1 / n) / a[k][k]
            a[k + 1:n + 2, k:k + 1].fill_(c)

        return a

    # handle target being both 1D tensor, and
    # target being 2D tensor (2D tensor means.nt: anything)
    def _transformTarget(self, target):
        assert target.dim() == 1
        nSamples = target.size(0)
        self._target.resize_(nSamples, self.nClasses)
        for i in range(nSamples):
            self._target[i].copy_(self.simplex[int(target[i])])

    def updateOutput(self, input, target):
        self._transformTarget(target)

        assert input.nelement() == self._target.nelement()
        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        self._backend.MSECriterion_updateOutput(
            self._backend.library_state,
            input,
            self._target,
            self.output_tensor,
            self.sizeAverage
        )
        self.output = self.output_tensor[0]
        return self.output

    def updateGradInput(self, input, target):
        assert input.nelement() == self._target.nelement()
        self._backend.MSECriterion_updateGradInput(
            self._backend.library_state,
            input,
            self._target,
            self.gradInput,
            self.sizeAverage
        )
        return self.gradInput

    def getPredictions(self, input):
        return torch.mm(input, self.simplex.t())

    def getTopPrediction(self, input):
        prod = self.getPredictions(input)
        _, maxs = prod.max(prod.ndimension() - 1)
        return maxs.view(-1)
