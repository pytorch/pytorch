import torch
from torch.nn.parameter import Parameter

from .module import Module


class Embedding(Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If given, pads the output with zeros whenever it encounters the index.
        max_norm (float, optional): If given, will renormalize the embeddings to always have a norm lesser than this
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the frequency of
                                                the words in the mini-batch.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)

    Shape:
        - Input: LongTensor `(N, W)`, N = mini-batch, W = number of indices to extract per mini-batch
        - Output: `(N, W, embedding_dim)`

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) =
         -1.0822  1.2522  0.2434
          0.8393 -0.6062 -0.3348
          0.6597  0.0350  0.0837
          0.5521  0.9447  0.0498

        (1 ,.,.) =
          0.6597  0.0350  0.0837
         -0.1527  0.0877  0.4260
          0.8393 -0.6062 -0.3348
         -0.8738 -0.9054  0.4281
        [torch.FloatTensor of size 2x4x3]

        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = Variable(torch.LongTensor([[0,2,0,5]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) =
          0.0000  0.0000  0.0000
          0.3452  0.4937 -0.9361
          0.0000  0.0000  0.0000
          0.0706 -2.1962 -0.6276
        [torch.FloatTensor of size 1x4x3]

    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.sparse = sparse

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1
        return self._backend.Embedding(
            padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse
        )(input, self.weight)

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class EmbeddingSum(Module):
    r"""Computes sums of 'bags' of embeddings, without instantiating the
    intermediate embeddings.

    For bags of constant length, nn.EmbeddingSum is equivalent to nn.Embedding
    followed by Sum(1), but it is more time- and memory-efficient on GPU.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If given, pads the output with zeros whenever it encounters the index.
        max_norm (float, optional): If given, will renormalize the embeddings to always have a norm lesser than this
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the frequency of
                                                the words in the dictionary.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)

    Shape:
        - Input: LongTensor `N`, N = number of embeddings to extract
        - Offsets: LongTensor `B`, B = number of bags. The values are the offsets in `input` for each bag, i.e. the cumsum of lengths.
        - Output: `(B, embedding_dim)`

    Examples:

    >>> # an Embedding module containing 10 tensors of size 3
    >>> embedding_sum = nn.EmbeddingSum(10, 3)
    >>> # a batch of 2 samples of 4 indices each
    >>> input = Variable(torch.LongTensor([1,2,4,5,4,3,2,9]))
    >>> offsets = Variable(torch.LongTensor([0,4]))
    >>> embedding_sum(input, offsets)

    Variable containing:
    -0.7296 -4.6926  0.3295
    -0.5186 -0.5631 -0.2792
    [torch.FloatTensor of size 2x3]
    """

    def __init__(self, num_embeddings, embedding_dim,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False):
        super(EmbeddingSum, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

    def forward(self, input, offsets):
        return self._backend.EmbeddingSum(
            self.max_norm, self.norm_type,
            self.scale_grad_by_freq
        )(input, offsets, self.weight)

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

# TODO: SparseLinear
