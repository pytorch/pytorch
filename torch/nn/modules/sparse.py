import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from .module import Module
from .. import functional as F


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
        sparse (boolean, optional): if ``True``, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for
                                    more details regarding sparse gradients.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)

    Shape:
        - Input: LongTensor `(N, W)`, N = mini-batch, W = number of indices to extract per mini-batch
        - Output: `(N, W, embedding_dim)`

    Notes:
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's `optim.SGD` (`cuda` and `cpu`),
        `optim.SparseAdam` (`cuda` and `cpu`) and `optim.Adagrad` (`cpu`)

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
        >>> embedding(input)

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
        [torch.FloatTensor of size (2,4,3)]

        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = torch.LongTensor([[0,2,0,5]])
        >>> embedding(input)

        (0 ,.,.) =
          0.0000  0.0000  0.0000
          0.3452  0.4937 -0.9361
          0.0000  0.0000  0.0000
          0.0706 -2.1962 -0.6276
        [torch.FloatTensor of size (1,4,3)]

    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)
        self.sparse = sparse

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

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

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as 'num_embeddings', second as 'embedding_dim'.
            freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``

        Examples::

            >> # FloatTensor containing pretrained weights
            >> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >> embedding = nn.Embedding.from_pretrained(weight)
            >> # Get embeddings for index 1
            >> input = torch.LongTensor([1])
            >> embedding(input)

             4.0000  5.1000  6.3000
            [torch.FloatTensor of size (1,3)]
        """
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(num_embeddings=rows, embedding_dim=cols, _weight=embeddings)
        embedding.weight.requires_grad = not freeze
        return embedding


class EmbeddingBag(Module):
    r"""Computes sums or means of 'bags' of embeddings, without instantiating the
    intermediate embeddings.

    For bags of constant length,
        * nn.EmbeddingBag with `mode=sum` is equivalent to nn.Embedding followed by `torch.sum(dim=1)`
        * with `mode=mean` is equivalent to nn.Embedding followed by `torch.mean(dim=1)`

    However, nn.EmbeddingBag is much more time and memory efficient than using a chain of these
    operations.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        max_norm (float, optional): If given, will renormalize the embeddings to always have a norm lesser than this
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the frequency of
                                                the words in the dictionary.
        mode (string, optional): 'sum' | 'mean'. Specifies the way to reduce the bag. Default: 'mean'
        sparse (boolean, optional): if ``True``, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for
                                    more details regarding sparse gradients.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)

    Inputs: input, offsets
        - **input** (N or BxN): LongTensor containing the indices of the embeddings
                                to extract. When `input` is 1D Tensor of shape `N`,
                                an `offsets` Tensor is given, that contains the
                                starting position of each new sequence in the
                                mini-batch.
        - **offsets** (B or None): LongTensor containing the starting positions of
                                   each sample in a mini-batch of variable length
                                   sequences. If `input` is 2D (BxN), then offsets
                                   does not need to be given, as the `input` is
                                   treated as a mini-batch of fixed length sequences
                                   of length `N` each.


    Shape:
        - Input: LongTensor `N`, N = number of embeddings to extract
                 (or) LongTensor `BxN`, B = number of sequences in mini-batch,
                                        N = number of embeddings per sequence
        - Offsets: LongTensor `B`, B = number of bags. The values are the
                   offsets in `input` for each bag, i.e. the cumsum of lengths.
                   Offsets is not given if Input is 2D `BxN` Tensor,
                   the input is considered to be of fixed-length sequences
        - Output: `(B, embedding_dim)`

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding_sum = nn.EmbeddingBag(10, 3, mode='sum')
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([1,2,4,5,4,3,2,9])
        >>> offsets = torch.LongTensor([0,4])
        >>> embedding_sum(input, offsets)

        -0.7296 -4.6926  0.3295
        -0.5186 -0.5631 -0.2792
        [torch.FloatTensor of size (2,3)]

    """

    def __init__(self, num_embeddings, embedding_dim,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 mode='mean', sparse=False):
        super(EmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.mode = mode
        self.sparse = sparse

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

    def forward(self, input, offsets=None):
        return F.embedding_bag(self.weight, input, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.mode, self.sparse)

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        s += ', mode={mode}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

# TODO: SparseLinear
