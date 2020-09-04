import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.quantized.modules.utils import hide_packed_params_repr
from torch.nn.quantized.modules.utils import _quantize_weight
from torch.quantization.qconfig import float_qparams_dynamic_qconfig
from typing import Optional

class EmbeddingPackedParams(torch.nn.Module):
    _version = 1

    def __init__(self, num_embeddings, embedding_dim, dtype=torch.quint8) -> None:
        super(EmbeddingPackedParams, self).__init__()
        self.dtype = dtype
        if self.dtype == torch.quint8:
            scales = torch.ones(num_embeddings, dtype=torch.float)
            zero_points = torch.ones(num_embeddings, dtype=torch.float)
            wq = torch._empty_per_channel_affine_quantized([num_embeddings, embedding_dim], scales=scales,
                                                           zero_points=zero_points,
                                                           axis=0, dtype=torch.quint8)
            self.set_weight(wq)
        else:
            raise RuntimeError('Unsupported dtype on dynamic quantized embedding_bag!')

    @torch.jit.export
    def set_weight(self, weight: Tensor) -> None:
        if self.dtype == torch.quint8:
            self._packed_weight = torch.ops.quantized.embedding_bag_prepack(weight)
        else:
            raise RuntimeError('Unsupported dtype on dynamic quantized embedding_bag!')


    @torch.jit.export
    def _weight(self):
        if self.dtype == torch.quint8:
            return torch.ops.quantized.embedding_bag_unpack(self._packed_weight)
        else:
            raise RuntimeError('Unsupported dtype on dynamic quantized embedding_bag!')

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
        version = local_metadata.get('version', None)
        self.dtype = state_dict[prefix + 'dtype']
        state_dict.pop(prefix + 'dtype')

        weight = state_dict[prefix + '_packed_weight']
        state_dict.pop(prefix + '_packed_weight')
        self.set_weight(weight)

        super(EmbeddingPackedParams, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                                 missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        return self._weight().__repr__()

class EmbeddingBag(torch.nn.Module):
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
        >>> m = nn.quantized.dynamic.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, mode='sum')
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        >>> offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        >>> output = m(indices, offsets)
        >>> print(output.size())
        torch.Size([5, 12]

    """
    _FLOAT_MODULE = nn.EmbeddingBag
    _version = 1

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 mode: str = 'sum', sparse: bool = False, _weight: Optional[Tensor] = None,
                 include_last_offset: bool = False, dtype=torch.quint8) -> None:
        super(EmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset

        if _weight is None:
            scales = torch.ones(num_embeddings, dtype=torch.float)
            zero_points = torch.ones(num_embeddings, dtype=torch.float)
            self.qweight = torch._empty_per_channel_affine_quantized([num_embeddings, embedding_dim],
                                                                     scales=scales, zero_points=zero_points,
                                                                     axis=0, dtype=torch.quint8)
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.qweight = _weight

        self._packed_params = EmbeddingPackedParams(num_embeddings, embedding_dim, dtype)
        self._packed_params.set_weight(self.qweight)

    def forward(self, indices: Tensor, offsets: Optional[Tensor] = None, per_sample_weights: Optional[Tensor] = None,
                compressed_indices_mapping: Optional[Tensor] = None) -> Tensor:
        return torch.ops.quantized.embedding_bag_byte(self._packed_params._packed_weight, indices, offsets, False, 0,
                                                      self.sparse, per_sample_weights, compressed_indices_mapping,
                                                      self.include_last_offset)

    def _get_name(self):
        return 'DynamicQuantizedEmbeddingBag'

    def __repr__(self):
        return hide_packed_params_repr(self, EmbeddingPackedParams)

    def extra_repr(self):
        extra_repr_str = 'num_embeddings={}, embedding_dim={}, dtype={}, qscheme={}, sparse={}'.format(
            self.num_embeddings, self.embedding_dim, self._packed_params.dtype, self.qweight.qscheme(), self.sparse
        )

        return extra_repr_str

    def set_weight(self, w: Tensor) -> None:
        self._packed_params.set_weight(w)

    def weight(self):
        return self._packed_params._weight()

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized embedding_bag module from a float module

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'nnqd.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'EmbeddingBag input float module must have qconfig defined'
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
        else:
            weight_observer = float_qparams_dynamic_qconfig.weight()

        dtype = weight_observer.dtype

        assert dtype == torch.quint8, 'The only supported dtype for nnqd.EmbeddingBag is torch.quint8'

        # Run the observer to calculate qparams.
        weight_observer(mod.weight)
        qweight = _quantize_weight(mod.weight.float(), weight_observer)

        # Create quantized EmbeddingBag module and pass in the quantized weight
        qembedding_bag = EmbeddingBag(mod.num_embeddings, mod.embedding_dim)
        qembedding_bag.set_weight(qweight)
        return qembedding_bag
