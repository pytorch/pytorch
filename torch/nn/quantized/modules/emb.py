r"""Quantized EmbeddingBag modules."""

import torch
import torch.nn as nn
from ... import functional as F


class EmbeddingBag(nn.Module):
    _FLOAT_MODULE = nn.EmbeddingBag

    def __init__(self, num_embeddings, embedding_dim,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 mode='mean', sparse=False, _weight=None):
        super(EmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.mode = mode
        self.sparse = sparse

        # TODO: per channel quantization
        self.weight = _weight

    def forward(self, input, offsets=None, per_sample_weights=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        return F.embedding_bag(input, self.weight, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.mode, self.sparse,
                               per_sample_weights)

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        weight_observer = mod.qconfig.weight()
        weight_observer(mod.weight)
        assert weight_observer.dtype == torch.qint8, 'Weight observer must have a dtype of qint8'
        wt_scale, wt_zp = weight_observer.calculate_qparams()
        assert len(wt_scale) == mod.weight.size()[0], 'only support per_channel_quantization now'
        qweight = torch.quantize_per_channel(mod.weight.float(), wt_scale, wt_zp, 0,
                                                    torch.qint8)
        qemb = cls(mod.num_embeddings, mod.embedding_dim, mod.max_norm, mod.norm_type,
                   mod.scale_grad_by_freq, mod.mode, mod.sparse, None)
        qemb.weight = qweight
        return qemb