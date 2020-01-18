from .module import Module
from .. import functional as F


class Relation(Module):
    def __init__(self, g, max_pairwise=None):
        r"""Applies an all-to-all pairwise relation function to the incoming
        data:

        .. math:: `y = \left(\sum_{i, j} g(x_i, x_j, e)\right)`

        Relation networks are modules which allow networks to reason over pairs
        of objects by taking features (dimension 3) from objects (dimension 2)
        and applying a function `g` to each possible pairwise combination of
        objects. The outputs of this operation are then summed, in order to be
        invariant to the ordering of the objects.

        Note that this does not include the final function `f` as described in
        "A simple neural network module for relational reasoning", as this can
        be a generic function/network and is not key to the pairwise
        decomposition.

        Given pairs of inputs, `g` must take in inputs of size
        `2 * in\_features`. To perform more powerful comparisons, relation
        networks can condition their operations on an embedding, where a single
        embedding vector (per batch) is concatenated to each pairwise set of
        features. `e` can be provided as an optional second argument in the
        forward pass of this module. If `e` is to be provided, `g` must take in
        inputs of size `2 * in\_features + embedding\_size`.

        Args:
            g: function/network applied to the pairs of objects x. May also
               take an embedding concatenated to each pair.
            max_pairwise: max number of pairwise calculations to perform
                          at once. Used to limit the max number of simultaneous
                          operations when there are many objects or `g` is
                          expensive to calculate.

        Shape:
            - Input: :math:`(N, objects, in\_features)` and optionally
                           `(N, embedding\_size)`
            - Output: :math:`(N, out\_features)`

        Examples::

            >>> # 60 is 2 * 30 (double the feature size per object)
            >>> g = nn.Linear(60, 20)
            >>> m = nn.Relation(g)
            >>> input = torch.randn(32, 5, 30)
            >>> output = m(input)
            >>> # 80 now includes the embedding size 20
            >>> g = nn.Linear(80, 20)
            >>> m = nn.Relation(g)
            >>> embedding = torch.randn(32, 20)
            >>> output = m(input, embedding)
        """
        super(Relation, self).__init__()
        self.g = g
        self.max_pairwise = max_pairwise

    def forward(self, x, e=None):
        return F.relation(x, self.g, e, self.max_pairwise)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.g) + ')'
