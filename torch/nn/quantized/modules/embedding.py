from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Optional, List  # noqa: F401
from torch.nn.quantized.modules.utils import hide_packed_params_repr
from torch.nn.quantized.modules.utils import _quantize_weight

class EmbeddingPackedParams(torch.nn.Module):
    _version = 1

    def __init__(self, num_embeddings, embedding_dim, dtype=torch.quint8):
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
            raise RuntimeError('Unsupported dtype on quantized embedding!')

    @torch.jit.export
    def set_weight(self, weight):
        # type: (torch.Tensor) -> None
        if self.dtype == torch.quint8:
            self._packed_weight = torch.ops.quantized.embedding_bag_prepack(weight)
        else:
            raise RuntimeError('Unsupported dtype on quantized embedding!')


    @torch.jit.export
    def _weight(self):
        if self.dtype == torch.quint8:
            return torch.ops.quantized.embedding_bag_unpack(self._packed_weight)
        else:
            raise RuntimeError('Unsupported dtype on quantized embedding!')

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
        torch.Size([9, 12]

    """
    _FLOAT_MODULE = nn.Embedding
    _version = 1

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, dtype=torch.quint8) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparse = sparse

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

    def forward(self, indices: Tensor) -> Tensor:
        return torch.ops.quantized.embedding_byte(self._packed_params._packed_weight, indices, self.sparse)

    def _get_name(self):
        return 'QuantizedEmbedding'

    def __repr__(self):
        return hide_packed_params_repr(self, EmbeddingPackedParams)

    def extra_repr(self):
        extra_repr_str = 'num_embeddings={}, embedding_dim={}, dtype={}, qscheme={}, sparse={}'.format(
            self.num_embeddings, self.embedding_dim, self._packed_params.dtype, self.qweight.qscheme(), self.sparse
        )

        return extra_repr_str

    def set_weight(self, w):
        # type: (torch.Tensor) -> None
        self._packed_params.set_weight(w)

    def weight(self):
        return self._packed_params._weight()

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized embedding module from a float module

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'nnqd.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Embedding input float module must have qconfig defined'
        from torch.quantization.qconfig import float_qparams_dynamic_qconfig
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
        else:
            weight_observer = float_qparams_dynamic_qconfig.weight()

        dtype = weight_observer.dtype

        assert dtype == torch.quint8, 'The only supported dtype for nnq.Embedding is torch.quint8'

        # Run the observer to calculate qparams.
        weight_observer(mod.weight)
        qweight = _quantize_weight(mod.weight.float(), weight_observer)

        # Create quantized Embedding module and pass in the quantized weight
        qembedding = Embedding(mod.num_embeddings, mod.embedding_dim)
        qembedding.set_weight(qweight)
        return qembedding
