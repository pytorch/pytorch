""" Define Base class as well as some crowd favorites """
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Optional, Union
from warnings import warn

import torch
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    SDPAParams,
)
from torch.nn.functional import scaled_dot_product_attention
from torch.utils import _pytree as pytree


def input_requires_grad(*tensors: torch.Tensor) -> bool:
    return any(t.requires_grad for t in tensors)


def materialize_if_needed(
    bias: "AttnBias", device: Optional[torch.device] = None
) -> Union[torch.Tensor, "AttnBias"]:
    if bias.needs_materialization():
        return bias.materialize(device)
    return bias


class AttnBias(ABC):
    """Abstract base class for attention biases"""

    @abstractmethod
    def materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        raise NotImplementedError("This is an abstract base class")

    @abstractmethod
    def needs_materialization(self) -> bool:
        raise NotImplementedError("This is an abstract base class")

    @staticmethod
    @abstractmethod
    def dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask,
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
    ) -> torch.Tensor:
        raise NotImplementedError("This is an abstract base class")


class TensorBias(AttnBias):
    """A bias that is a tensor"""

    def __init__(self, bias: torch.Tensor):
        self.bias = bias

    def materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.bias

    def needs_materialization(self) -> bool:
        return True

    @staticmethod
    def dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "AttnBias",
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "TensorBias requires materialization, so this should never be called!"
        )

    def __repr__(self) -> str:
        return f"TensorBias(bias={self.bias})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func != torch.nn.functional.scaled_dot_product_attention:
            return NotImplemented
        args = pytree.tree_map_only(
            TensorBias, lambda x: materialize_if_needed(x), args
        )
        kwargs = pytree.tree_map_only(
            TensorBias, lambda x: materialize_if_needed(x), kwargs
        )
        return func(*args, **kwargs)


class LambdaBias(AttnBias):
    """A bias that is a function"""

    def __init__(self, bias_fn):
        self.bias_fn = bias_fn

    def materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.bias_fn()

    def needs_materialization(self) -> bool:
        return False

    @staticmethod
    def dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "LambdaBias",
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
    ) -> torch.Tensor:
        raise NotImplementedError("TODO FIGURE OUT!")


class CausalVariant(IntEnum):
    """Enum for causal variants"""

    UPPER_LEFT = 1
    LOWER_RIGHT = 2


class CausalBias(AttnBias):
    """A bias representing causal attention patterns"""

    def __init__(self, variant: CausalVariant, seq_len_q: int, seq_len_kv: int):
        assert isinstance(variant, CausalVariant)
        self.variant = variant
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        if seq_len_q > seq_len_kv and variant == CausalVariant.LOWER_RIGHT:
            warn(
                "Lower right causal bias will produce NaNs in the output when seq_len_q > seq_len_kv!"
            )

    def _upper_left(self, device: torch.device) -> torch.Tensor:
        """Upper left causal bias"""
        return torch.tril(
            torch.ones(self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool)
        )

    def _lower_right(self, device: torch.device) -> torch.Tensor:
        """Lower right causal bias"""
        diagonal_offset = self.seq_len_kv - self.seq_len_q
        return torch.tril(
            torch.ones(
                self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool
            ),
            diagonal=diagonal_offset,
        )

    def materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")
        if self.variant == CausalVariant.UPPER_LEFT:
            return self._upper_left(device)
        elif self.variant == CausalVariant.LOWER_RIGHT:
            return self._lower_right(device)

    def needs_materialization(self) -> bool:
        return False

    @staticmethod
    def dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "CausalBias",
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
    ) -> torch.Tensor:
        if is_causal:
            raise ValueError("CausalBias should not be used with causal=True")

        if attn_mask.seq_len_q == attn_mask.seq_len_kv:
            return scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
                scale=scale,
            )
        if attn_mask.variant == CausalVariant.UPPER_LEFT:
            return scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
                scale=scale,
            )
        elif attn_mask.variant == CausalVariant.LOWER_RIGHT:
            sdpa_params = SDPAParams(query, key, value, None, dropout_p, is_causal)
            if can_use_flash_attention(sdpa_params):
                needs_slicing = query.size(-1) % 8 != 0
                og_head_size = query.size(-1)
                if needs_slicing:
                    query = torch.nn.functional.pad(query, (0, 8 - query.size(-1) % 8))
                    key = torch.nn.functional.pad(key, (0, 8 - key.size(-1) % 8))
                    value = torch.nn.functional.pad(value, (0, 8 - value.size(-1) % 8))
                out = torch.ops.aten._scaled_dot_product_flash_attention(
                    query,
                    key,
                    value,
                    dropout_p,
                    is_causal=True,
                    return_debug_mask=False,
                    scale=scale,
                )[0]
                if needs_slicing:
                    out = out[..., :og_head_size]
                return out
            if can_use_efficient_attention(sdpa_params):
                compute_log_sumexp = False
                if input_requires_grad(query, key, value):
                    compute_log_sumexp = True
                return torch.ops.aten._efficient_attention_forward(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    bias=None,
                    cu_seqlens_q=None,
                    cu_seqlens_k=None,
                    max_seqlen_q=None,
                    dropout_p=dropout_p,
                    custom_mask_type=int(attn_mask.variant),
                    compute_log_sumexp=compute_log_sumexp,
                    scale=scale,
                    causal_diagonal=None,
                    seqlen_k=None,
                )[0].transpose(1, 2)
            else:
                # TODO This will warn with the reason why we cant use efficient attention
                # Should this default to on?
                can_use_efficient_attention(sdpa_params, True)
                # We cant use efficient attention the only support for lower right is via materialization
                return scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask.materialize(query.device),
                    dropout_p=dropout_p,
                    is_causal=False,
                    scale=scale,
                )
        else:
            raise ValueError("Invalid causal variant")

    def __repr__(self) -> str:
        return f"CausalBias(variant={self.variant.name}, seq_len_q={self.seq_len_q}, seq_len_kv={self.seq_len_kv})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func != torch.nn.functional.scaled_dot_product_attention:
            return NotImplemented
        return cls.dispatch(*args, **kwargs)
