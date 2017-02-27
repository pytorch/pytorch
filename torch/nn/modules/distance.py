import torch
from .module import Module
from .. import functional as F
# TODO: Cosine


class CosineSimilarity(Module):
    """
    Computes the batchwise cosine similarity between vectors v1,v2:
    
        .. math ::
            cossim(v_1,v_2) = \frac{v_1 * v_2}{\left| v_1 \right| * \left| v_2 \right|}
        
        Inputs: x
            - **x** of size (2, batch, size): Tensor containing the two input batches x1,x2 = x.

        Outputs: output
            - **output** of size (batch,): Tensor containing the batch-wise similarities between the batched inputs x1,x2.

    """

    def __init__(self):
        super(CosineSimilarity, self).__init__()
        self.eps = 1e-12

    def forward(self, x):
        return F.cosine_similarity(x,self.eps)


# TODO: Euclidean
# TODO: WeightedEuclidean
# TODO: PairwiseDistance
