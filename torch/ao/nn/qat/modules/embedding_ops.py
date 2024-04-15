import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Embedding', 'EmbeddingBag']

class Embedding(nn.Embedding):
    r"""
    An embedding bag module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
    for documentation.

    Similar to `torch.nn.Embedding`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Embedding

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 sparse=False, _weight=None, device=None, dtype=None, qconfig=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight,
                         **factory_kwargs)
        assert qconfig, 'qconfig must be provided for QAT module'
        assert qconfig.weight().qscheme == torch.per_channel_affine_float_qparams, \
            'Embedding weights requires a qscheme of torch.per_channel_affine_float_qparams Got ' + \
            str(qconfig.weight().qscheme)
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input) -> Tensor:
        return F.embedding(input, self.weight_fake_quant(self.weight), self.padding_idx,
                           self.max_norm, self.norm_type, self.scale_grad_by_freq,
                           self.sparse)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        weight_qscheme = mod.qconfig.weight().qscheme  # type: ignore[union-attr, operator]
        assert weight_qscheme == torch.per_channel_affine_float_qparams, \
            'Embedding weights requires a qscheme of torch.per_channel_affine_float_qparams Got ' + \
            str(weight_qscheme)

        qconfig = mod.qconfig
        qat_embedding_bag = cls(mod.num_embeddings, mod.embedding_dim, mod.padding_idx,
                                mod.max_norm, mod.norm_type, mod.scale_grad_by_freq,
                                mod.sparse, mod.weight, qconfig=qconfig)

        return qat_embedding_bag

    def to_float(self):
        embedding_bag = torch.nn.Embedding(self.num_embeddings, self.embedding_dim, self.padding_idx,
                                           self.max_norm, self.norm_type, self.scale_grad_by_freq,
                                           self.sparse, None)
        embedding_bag.weight = torch.nn.Parameter(self.weight.detach())
        embedding_bag.train(self.training)
        return embedding_bag

class EmbeddingBag(nn.EmbeddingBag):
    r"""
    An embedding bag module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.EmbeddingBag`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag
    for documentation.

    Similar to `torch.nn.EmbeddingBag`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.EmbeddingBag

    def __init__(self, num_embeddings, embedding_dim, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, mode='mean',
                 sparse=False, _weight=None, include_last_offset=False,
                 padding_idx=None, qconfig=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_embeddings, embedding_dim, max_norm, norm_type,
                         scale_grad_by_freq, mode, sparse, _weight,
                         include_last_offset, padding_idx, **factory_kwargs)
        assert qconfig, 'qconfig must be provided for QAT module'
        assert qconfig.weight().qscheme == torch.per_channel_affine_float_qparams, \
            'Embedding Bag weights requires a qscheme of torch.per_channel_affine_float_qparams Got ' + \
            str(qconfig.weight().qscheme)
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input, offsets=None, per_sample_weights=None) -> Tensor:
        return F.embedding_bag(input, self.weight_fake_quant(self.weight), offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.mode, self.sparse,
                               per_sample_weights, self.include_last_offset,
                               self.padding_idx)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        weight_qscheme = mod.qconfig.weight().qscheme  # type: ignore[union-attr, operator]
        assert weight_qscheme == torch.per_channel_affine_float_qparams, \
            'Embedding Bag weights requires a qscheme of torch.per_channel_affine_float_qparams Got ' + \
            str(weight_qscheme)

        qconfig = mod.qconfig
        qat_embedding_bag = cls(mod.num_embeddings, mod.embedding_dim, mod.max_norm, mod.norm_type,
                                mod.scale_grad_by_freq, mod.mode, mod.sparse, mod.weight,
                                mod.include_last_offset, mod.padding_idx, qconfig=qconfig)

        return qat_embedding_bag

    def to_float(self):
        embedding_bag = torch.nn.EmbeddingBag(self.num_embeddings, self.embedding_dim, self.max_norm,
                                              self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse,
                                              None, self.include_last_offset, self.padding_idx)
        embedding_bag.weight = torch.nn.Parameter(self.weight.detach())
        embedding_bag.train(self.training)
        return embedding_bag
