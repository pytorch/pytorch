import torch
from torch.autograd import Variable

from .module import Module


class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.
    Args:
        num_embeddings: size of the dictionary of embeddings
        embedding_dim: the size of each embedding vector
        padding_idx: If given, pads the output with zeros whenever it encounters the index. Default: -1
        max_norm: If given, will renormalize the embeddings to always have a norm lesser than this Default: None
        norm_type: The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq: if given, this will scale gradients by the frequency of the words in the dictionary.
    Input Shape: [ *, * ] : Input is a 2D mini_batch LongTensor of m x n indices to extract from the Embedding dictionary
    Output Shape:[ * , *, * ]  : Output shape = m x n x embedding_dim
    Examples:
        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.Tensor([[1,2,4,5],[4,3,2,10]])
        >>> print(embedding(input))
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=-1,
            max_norm=None, norm_type=2, scale_grad_by_freq=False):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        super(Embedding, self).__init__(
            weight=Variable(torch.Tensor(num_embeddings, embedding_dim))
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

    def forward(self, input):
        return self._backend.Embedding(self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq)(input, self.weight)


# TODO: SparseLinear

