import torch
from .module import Module
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
        assert len(x) == 2,"Input needs to be of size (2, batch, dim)"
        x1, x2 = x
        w12 = torch.sum(x1*x2,1)
        w1 = torch.norm(x1, 2, 1)
        w2 = torch.norm(x2, 2, 1)
        return (w12 / (w1 * w2) + self.eps).squeeze()


# TODO: Euclidean
# TODO: WeightedEuclidean
# TODO: PairwiseDistance
