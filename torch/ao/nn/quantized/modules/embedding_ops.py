import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Optional, List  # noqa: F401

from .utils import hide_packed_params_repr
from .utils import _quantize_weight

__all__ = ['EmbeddingPackedParams', 'Embedding', 'EmbeddingBag']

class EmbeddingPackedParams(torch.nn.Module):
    _version = 1

    def __init__(self, num_embeddings, embedding_dim, dtype=torch.quint8):
        super(EmbeddingPackedParams, self).__init__()
        self.dtype = dtype
        if self.dtype in [torch.quint8, torch.quint4x2]:
            scales = torch.ones(num_embeddings, dtype=torch.float)
            zero_points = torch.zeros(num_embeddings, dtype=torch.float)
            wq = torch._empty_per_channel_affine_quantized([num_embeddings, embedding_dim], scales=scales,
                                                           zero_points=zero_points,
                                                           axis=0, dtype=self.dtype)
            self.set_weight(wq)
        else:
            raise NotImplementedError(f'Unsupported dtype on quantized embedding! Supports quint8 and quint4x2. Got dtype: {dtype}')

    @torch.jit.export
    def set_weight(self, weight: torch.Tensor) -> None:
        if self.dtype in [torch.quint8, torch.quint4x2]:
            self._packed_weight = torch.ops.quantized.embedding_bag_prepack(weight)
        else:
            raise NotImplementedError('Unsupported dtype for quantized embedding prepack! Supports quint8 and quint4x2.')


    @torch.jit.export
    def _weight(self):
        if self.dtype in [torch.quint8, torch.quint4x2]:
            return torch.ops.quantized.embedding_bag_unpack(self._packed_weight)
        else:
            raise NotImplementedError('Unsupported dtype for quantized embedding unpack! Supports quint8 and quint4x2.')

    def forward(self, x):
        return x

    # Version 1
    #   self
    #   |--- _packed_weight : Tensor representing weight of EmbeddingPackedParamsBase
    #   |--- dtype : torch.dtype

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(EmbeddingPackedParams, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'dtype'] = self.dtype
        destination[prefix + '_packed_weight'] = self._weight()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.dtype = state_dict[prefix + 'dtype']
        state_dict.pop(prefix + 'dtype')

        weight = state_dict[prefix + '_packed_weight']
        state_dict.pop(prefix + '_packed_weight')
        self.set_weight(weight)

        super(EmbeddingPackedParams, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                                 missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        return self._weight().__repr__()

class Embedding(torch.nn.Module):
    r"""
    A quantized Embedding module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding for documentation.

    Similar to :class:`~torch.nn.Embedding`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{num\_embeddings}, \text{embedding\_dim})`.

    Examples::
        >>> m = nn.quantized.Embedding(num_embeddings=10, embedding_dim=12)
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8])
        >>> output = m(indices)
        >>> print(output.size())
        torch.Size([9, 12])

    """
    _version = 1

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, dtype=torch.quint8) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        if _weight is None:
            scales = torch.ones(num_embeddings, dtype=torch.float)
            zero_points = torch.zeros(num_embeddings, dtype=torch.float)
            qweight = torch._empty_per_channel_affine_quantized([num_embeddings, embedding_dim],
                                                                scales=scales, zero_points=zero_points,
                                                                axis=0, dtype=torch.quint8)
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            qweight = _weight

        self._packed_params = EmbeddingPackedParams(num_embeddings, embedding_dim, dtype)
        self._packed_params.set_weight(qweight)

    def forward(self, indices: Tensor) -> Tensor:
        if self.dtype == torch.quint4x2:
            return torch.ops.quantized.embedding_4bit(self._packed_params._packed_weight, indices)
        else:
            return torch.ops.quantized.embedding_byte(self._packed_params._packed_weight, indices)

    def _get_name(self):
        return 'QuantizedEmbedding'

    def __repr__(self):
        return hide_packed_params_repr(self, EmbeddingPackedParams)

    def extra_repr(self):
        extra_repr_str = 'num_embeddings={}, embedding_dim={}, dtype={}, qscheme={}'.format(
            self.num_embeddings, self.embedding_dim, self._packed_params.dtype, self.weight().qscheme()
        )

        return extra_repr_str

    def set_weight(self, w: torch.Tensor) -> None:
        self._packed_params.set_weight(w)

    def weight(self):
        return self._packed_params._weight()

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized embedding module from a float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by user
        """
        if hasattr(mod, 'weight_fake_quant'):
            assert type(mod) == torch.ao.nn.qat.Embedding, 'nnq.' + cls.__name__ + '.from_float ' + \
                'with fake quant only works for ' + torch.ao.nn.qat.Embedding.__name__
            weight_observer = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            assert type(mod) == nn.Embedding, 'nnq.' + cls.__name__ + '.from_float only works for ' + \
                nn.Embedding.__name__
            assert hasattr(mod, 'qconfig'), 'Embedding input float module must have qconfig defined'
            from torch.ao.quantization import float_qparams_weight_only_qconfig
            if mod.qconfig is not None and mod.qconfig.weight is not None:  # type: ignore[union-attr]
                weight_observer = mod.qconfig.weight()  # type: ignore[union-attr, operator]
            else:
                weight_observer = float_qparams_weight_only_qconfig.weight()

        dtype = weight_observer.dtype
        is_float_qparams_qconfig = weight_observer.qscheme == torch.per_channel_affine_float_qparams
        assert is_float_qparams_qconfig, \
            'Embedding quantization is only supported with float_qparams_weight_only_qconfig.'

        assert dtype == torch.quint8 or dtype == torch.quint4x2, \
            f'The only supported dtype for nnq.Embedding is torch.quint8 and torch.quint4x2, got {dtype}'

        # Run the observer to calculate qparams.
        weight_observer(mod.weight)
        qweight = _quantize_weight(mod.weight.float(), weight_observer)

        # Create quantized Embedding module and pass in the quantized weight
        qembedding = Embedding(mod.num_embeddings, mod.embedding_dim)
        qembedding.set_weight(qweight)
        return qembedding

    @classmethod
    def from_reference(cls, ref_embedding):
        qembedding = cls(
            ref_embedding.num_embeddings,
            ref_embedding.embedding_dim,
            ref_embedding.padding_idx,
            ref_embedding.max_norm,
            ref_embedding.norm_type,
            ref_embedding.scale_grad_by_freq,
            ref_embedding.sparse,
            ref_embedding.get_quantized_weight(),
            ref_embedding.weight_dtype,
        )
        return qembedding

class EmbeddingBag(Embedding):
    r"""
    A quantized EmbeddingBag module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.EmbeddingBag`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.EmbeddingBag for documentation.

    Similar to :class:`~torch.nn.EmbeddingBag`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{num\_embeddings}, \text{embedding\_dim})`.

    Examples::
        >>> m = nn.quantized.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, mode='sum')
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        >>> offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        >>> output = m(indices, offsets)
        >>> print(output.size())
        torch.Size([5, 12])

    """
    _version = 1

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 mode: str = 'sum', sparse: bool = False, _weight: Optional[Tensor] = None,
                 include_last_offset: bool = False, dtype=torch.quint8) -> None:
        super(EmbeddingBag, self).__init__(num_embeddings, embedding_dim, _weight=_weight, dtype=dtype)

        self.mode = mode
        self.pruned_weights = False
        self.include_last_offset = include_last_offset
        self.dtype = dtype

    def forward(self, indices: Tensor, offsets: Optional[Tensor] = None, per_sample_weights: Optional[Tensor] = None,
                compressed_indices_mapping: Optional[Tensor] = None) -> Tensor:
        if self.dtype == torch.quint4x2:
            return torch.ops.quantized.embedding_bag_4bit(self._packed_params._packed_weight, indices, offsets, False, 0,
                                                          self.pruned_weights, per_sample_weights, compressed_indices_mapping,
                                                          self.include_last_offset)
        else:
            return torch.ops.quantized.embedding_bag_byte(self._packed_params._packed_weight, indices, offsets, False, 0,
                                                          self.pruned_weights, per_sample_weights, compressed_indices_mapping,
                                                          self.include_last_offset)

    def _get_name(self):
        return 'QuantizedEmbeddingBag'

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized embedding_bag module from a float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by user
        """
        if hasattr(mod, 'weight_fake_quant'):
            weight_observer = mod.weight_fake_quant
        else:
            assert type(mod) == nn.EmbeddingBag, 'nnq.' + cls.__name__ + '.from_float only works for ' + \
                nn.EmbeddingBag.__name__
            assert hasattr(mod, 'qconfig'), 'EmbeddingBag input float module must have qconfig defined'
            from torch.ao.quantization.qconfig import float_qparams_weight_only_qconfig
            if mod.qconfig is not None and mod.qconfig.weight is not None:  # type: ignore[union-attr]
                weight_observer = mod.qconfig.weight()  # type: ignore[union-attr, operator]
            else:
                weight_observer = float_qparams_weight_only_qconfig.weight()

        dtype = weight_observer.dtype
        is_float_qparams_qconfig = weight_observer.qscheme == torch.per_channel_affine_float_qparams
        assert is_float_qparams_qconfig, \
            'EmbeddingBag quantization is only supported with float_qparams_weight_only_qconfig.'

        assert dtype == torch.quint8 or dtype == torch.quint4x2, \
            f'The only supported dtype for nnq.EmbeddingBag is torch.quint8 and torch.quint4x2, got {dtype}'

        # Run the observer to calculate qparams.
        weight_observer(mod.weight)
        qweight = _quantize_weight(mod.weight.float(), weight_observer)

        # Create quantized EmbeddingBag module and pass in the quantized weight
        qembedding_bag = EmbeddingBag(mod.num_embeddings, mod.embedding_dim, dtype=dtype)
        qembedding_bag.set_weight(qweight)
        return qembedding_bag

    @classmethod
    def from_reference(cls, ref_embedding_bag):
        qembedding_bag = cls(
            ref_embedding_bag.num_embeddings,
            ref_embedding_bag.embedding_dim,
            ref_embedding_bag.max_norm,
            ref_embedding_bag.norm_type,
            ref_embedding_bag.scale_grad_by_freq,
            ref_embedding_bag.mode,
            ref_embedding_bag.sparse,
            ref_embedding_bag.get_quantized_weight(),
            ref_embedding_bag.include_last_offset,
            ref_embedding_bag.weight_dtype,
        )
        return qembedding_bag
