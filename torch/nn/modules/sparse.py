# mypy: allow-untyped-defs
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter

from .module import Module


__all__ = ["Embedding", "EmbeddingBag"]


class Embedding(Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad". For a newly constructed Embedding,
                                     the embedding vector at :attr:`padding_idx` will default to all zeros,
                                     but can be updated to another value to be used as the padding vector.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                 See Notes for more details regarding sparse gradients.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`

    Shape:
        - Input: :math:`(*)`, IntTensor or LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`

    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)

    .. note::
        When :attr:`max_norm` is not ``None``, :class:`Embedding`'s forward method will modify the
        :attr:`weight` tensor in-place. Since tensors needed for gradient computations cannot be
        modified in-place, performing a differentiable operation on ``Embedding.weight`` before
        calling :class:`Embedding`'s forward method requires cloning ``Embedding.weight`` when
        :attr:`max_norm` is not ``None``. For example::

            n, d, m = 3, 5, 7
            embedding = nn.Embedding(n, d, max_norm=True)
            W = torch.randn((m, d), requires_grad=True)
            idx = torch.tensor([1, 2])
            a = embedding.weight.clone() @ W.t()  # weight must be cloned for this to be differentiable
            b = embedding(idx) @ W.t()  # modifies weight in-place
            out = (a.unsqueeze(0) + b.unsqueeze(1))
            loss = out.sigmoid().prod()
            loss.backward()

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])


        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = torch.LongTensor([[0, 2, 0, 5]])
        >>> embedding(input)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.1535, -2.0309,  0.9315],
                 [ 0.0000,  0.0000,  0.0000],
                 [-0.1655,  0.9897,  0.0635]]])

        >>> # example of changing `pad` vector
        >>> padding_idx = 0
        >>> embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
        >>> embedding.weight
        Parameter containing:
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7895, -0.7089, -0.0364],
                [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
        >>> with torch.no_grad():
        ...     embedding.weight[padding_idx] = torch.ones(3)
        >>> embedding.weight
        Parameter containing:
        tensor([[ 1.0000,  1.0000,  1.0000],
                [-0.7895, -0.7089, -0.0364],
                [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    freeze: bool
    sparse: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
                requires_grad=not _freeze,
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = Parameter(_weight, requires_grad=not _freeze)

        self.sparse = sparse

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze=True,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        r"""Create Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (bool, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                         therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                         i.e. it remains as a fixed "pad".
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (bool, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert (
            embeddings.dim() == 2
        ), "Embeddings parameter is expected to be 2-dimensional"
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            _freeze=freeze,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        return embedding


class EmbeddingBag(Module):
    r"""Compute sums or means of 'bags' of embeddings, without instantiating the intermediate embeddings.

    For bags of constant length, no :attr:`per_sample_weights`, no indices equal to :attr:`padding_idx`,
    and with 2D inputs, this class

        * with ``mode="sum"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.sum(dim=1)``,
        * with ``mode="mean"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.mean(dim=1)``,
        * with ``mode="max"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.max(dim=1)``.

    However, :class:`~torch.nn.EmbeddingBag` is much more time and memory efficient than using a chain of these
    operations.

    EmbeddingBag also supports per-sample weights as an argument to the forward
    pass. This scales the output of the Embedding before performing a weighted
    reduction as specified by ``mode``. If :attr:`per_sample_weights` is passed, the
    only supported ``mode`` is ``"sum"``, which computes a weighted sum according to
    :attr:`per_sample_weights`.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (bool, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (str, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 ``"sum"`` computes the weighted sum, taking :attr:`per_sample_weights`
                                 into consideration. ``"mean"`` computes the average of the values
                                 in the bag, ``"max"`` computes the max value over each bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor. See
                                 Notes for more details regarding sparse gradients. Note: this option is not
                                 supported when ``mode="max"``.
        include_last_offset (bool, optional): if ``True``, :attr:`offsets` has one additional element, where the last element
                                      is equivalent to the size of `indices`. This matches the CSR format.
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the
                                     gradient; therefore, the embedding vector at :attr:`padding_idx` is not updated
                                     during training, i.e. it remains as a fixed "pad". For a newly constructed
                                     EmbeddingBag, the embedding vector at :attr:`padding_idx` will default to all
                                     zeros, but can be updated to another value to be used as the padding vector.
                                     Note that the embedding vector at :attr:`padding_idx` is excluded from the
                                     reduction.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape `(num_embeddings, embedding_dim)`
                         initialized from :math:`\mathcal{N}(0, 1)`.

    Examples::

        >>> # an EmbeddingBag module containing 10 tensors of size 3
        >>> embedding_sum = nn.EmbeddingBag(10, 3, mode='sum')
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
        >>> offsets = torch.tensor([0, 4], dtype=torch.long)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> embedding_sum(input, offsets)
        tensor([[-0.8861, -5.4350, -0.0523],
                [ 1.1306, -2.5798, -1.0044]])

        >>> # Example with padding_idx
        >>> embedding_sum = nn.EmbeddingBag(10, 3, mode='sum', padding_idx=2)
        >>> input = torch.tensor([2, 2, 2, 2, 4, 3, 2, 9], dtype=torch.long)
        >>> offsets = torch.tensor([0, 4], dtype=torch.long)
        >>> embedding_sum(input, offsets)
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7082,  3.2145, -2.6251]])

        >>> # An EmbeddingBag can be loaded from an Embedding like so
        >>> embedding = nn.Embedding(10, 3, padding_idx=2)
        >>> embedding_sum = nn.EmbeddingBag.from_pretrained(
                embedding.weight,
                padding_idx=embedding.padding_idx,
                mode='sum')
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "mode",
        "sparse",
        "include_last_offset",
        "padding_idx",
    ]

    num_embeddings: int
    embedding_dim: int
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    mode: str
    sparse: bool
    include_last_offset: bool
    padding_idx: Optional[int]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        include_last_offset: bool = False,
        padding_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        if _weight is None:
            self.weight = Parameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = Parameter(_weight)
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(
        self,
        input: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of EmbeddingBag.

        Args:
            input (Tensor): Tensor containing bags of indices into the embedding matrix.
            offsets (Tensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines
                the starting index position of each bag (sequence) in :attr:`input`.
            per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
                to indicate all weights should be taken to be ``1``. If specified, :attr:`per_sample_weights`
                must have exactly the same shape as input and is treated as having the same
                :attr:`offsets`, if those are not ``None``. Only supported for ``mode='sum'``.

        Returns:
            Tensor output shape of `(B, embedding_dim)`.

        .. note::

            A few notes about ``input`` and ``offsets``:

            - :attr:`input` and :attr:`offsets` have to be of the same type, either int or long

            - If :attr:`input` is 2D of shape `(B, N)`, it will be treated as ``B`` bags (sequences)
              each of fixed length ``N``, and this will return ``B`` values aggregated in a way
              depending on the :attr:`mode`. :attr:`offsets` is ignored and required to be ``None`` in this case.

            - If :attr:`input` is 1D of shape `(N)`, it will be treated as a concatenation of
              multiple bags (sequences).  :attr:`offsets` is required to be a 1D tensor containing the
              starting index positions of each bag in :attr:`input`. Therefore, for :attr:`offsets` of shape `(B)`,
              :attr:`input` will be viewed as having ``B`` bags. Empty bags (i.e., having 0-length) will have
              returned vectors filled by zeros.
        """
        return F.embedding_bag(
            input,
            self.weight,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        s += ", mode={mode}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        return s.format(**{k: repr(v) for k, v in self.__dict__.items()})

    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor,
        freeze: bool = True,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        include_last_offset: bool = False,
        padding_idx: Optional[int] = None,
    ) -> "EmbeddingBag":
        r"""Create EmbeddingBag instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the EmbeddingBag.
                First dimension is being passed to EmbeddingBag as 'num_embeddings', second as 'embedding_dim'.
            freeze (bool, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embeddingbag.weight.requires_grad = False``. Default: ``True``
            max_norm (float, optional): See module initialization documentation. Default: ``None``
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (bool, optional): See module initialization documentation. Default ``False``.
            mode (str, optional): See module initialization documentation. Default: ``"mean"``
            sparse (bool, optional): See module initialization documentation. Default: ``False``.
            include_last_offset (bool, optional): See module initialization documentation. Default: ``False``.
            padding_idx (int, optional): See module initialization documentation. Default: ``None``.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embeddingbag = nn.EmbeddingBag.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([[1, 0]])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embeddingbag(input)
            tensor([[ 2.5000,  3.7000,  4.6500]])
        """
        assert (
            embeddings.dim() == 2
        ), "Embeddings parameter is expected to be 2-dimensional"
        rows, cols = embeddings.shape
        embeddingbag = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            mode=mode,
            sparse=sparse,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )
        embeddingbag.weight.requires_grad = not freeze
        return embeddingbag
