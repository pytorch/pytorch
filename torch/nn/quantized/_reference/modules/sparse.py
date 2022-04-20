import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import ReferenceQuantizedModule
from typing import Optional, Dict, Any

class Embedding(nn.Embedding, ReferenceQuantizedModule):
    """ A reference quantized Embedding module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None,
                 device=None, dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight, device, dtype)
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        return "QuantizedEmbedding(Reference)"

    def forward(self, input: Tensor) -> Tensor:
        weight_quant_dequant = self.get_weight()
        return F.embedding(
            input, weight_quant_dequant, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    @classmethod
    def from_float(cls, mod, weight_qparams):
        return cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.padding_idx,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.sparse,
            mod.weight,
            mod.weight.device,
            mod.weight.dtype,
            weight_qparams)

class EmbeddingBag(nn.EmbeddingBag, ReferenceQuantizedModule):
    """ A reference quantized EmbeddingBag module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 mode: str = 'mean', sparse: bool = False, _weight: Optional[Tensor] = None,
                 include_last_offset: bool = False, padding_idx: Optional[int] = None,
                 device=None, dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(num_embeddings, embedding_dim, max_norm, norm_type,
                         scale_grad_by_freq, mode, sparse, _weight, include_last_offset,
                         padding_idx, device, dtype)
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        return "QuantizedEmbedding(Reference)"

    def forward(self, input: Tensor, offsets: Optional[Tensor] = None, per_sample_weights: Optional[Tensor] = None) -> Tensor:
        weight_quant_dequant = self.get_weight()
        return F.embedding_bag(input, weight_quant_dequant, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.mode, self.sparse,
                               per_sample_weights, self.include_last_offset,
                               self.padding_idx)

    @classmethod
    def from_float(cls, mod, weight_qparams):
        return cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.mode,
            mod.sparse,
            mod.weight,
            mod.include_last_offset,
            mod.padding_idx,
            mod.weight.device,
            mod.weight.dtype,
            weight_qparams
        )
