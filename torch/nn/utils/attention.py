""" Define Base class as well as some crowd favorites """
from abc import ABC, abstractmethod
from enum import auto, IntEnum
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


__all__ = ["AttnBias", "TensorBias", "CausalVariant", "CausalBias"]


def _input_requires_grad(*tensors: torch.Tensor) -> bool:
    """Returns True if any of the tensors requires grad"""
    return any(t.requires_grad for t in tensors)


def _materialize_if_needed(
    bias: "AttnBias", device: Optional[torch.device] = None
) -> Union[torch.Tensor, "AttnBias"]:
    if bias.needs_materialization():
        return bias.materialize(device)
    return bias


def _any_need_materialization(biases: "AttnBias") -> bool:
    """Returns True if any of the biases need materialization"""

    def needs_to_materialize(x):
        return isinstance(x, AttnBias) and x.needs_materialization()

    return any(map(needs_to_materialize, pytree.tree_leaves(biases)))


def _postprocess_flash_output(inpt_tensor: torch.Tensor, og_size: int) -> torch.Tensor:
    """Handles the unpad of the last dimension"""
    if inpt_tensor.size(-1) != og_size:
        return inpt_tensor[..., :og_size]
    return inpt_tensor


class AttnBias(ABC):
    """Abstract base class for attention biases"""

    @abstractmethod
    def materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Materialize the bias to a tensor for use with the default scaled_dot_product_attention function.
        This method is called when 'needs_materialization' returns True, converting the abstract bias
        representation into a concrete tensor form.

        This method must be implemented by any subclass of AttnBias, tailored to the specific bias representation.

        Parameters:
        - device: The device on which to materialize the tensor. Defaults to CPU if None is provided.

        Returns:
        - torch.Tensor: The materialized bias as a tensor.

        Raises:
        - NotImplementedError: If not implemented in a subclass.

        Note:
        - Implementers should ensure that the returned tensor is compatible with the expected input format of the
        scaled_dot_product_attention function, both in type and shape.
        """
        raise NotImplementedError("This is an abstract base class")

    @abstractmethod
    def needs_materialization(self) -> bool:
        """
        Determine whether the bias needs to be materialized as a tensor for the computation of scaled_dot_product_attention.

        This method should be implemented in any subclass of AttnBias to indicate whether the bias representation
        requires conversion (materialization) to a tensor form before being used in attention calculations.

        Returns:
        - bool: True if the bias needs to be materialized into a tensor before use; False otherwise.

        Raises:
        - NotImplementedError: If not implemented in a subclass.
        """
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
        """
        Determine how to compute the scaled dot product attention when the attn_bias may not require materialization.

        This method should be implemented in subclasses to define the specific attention computation approach
        when the bias can be used directly without converting it into a tensor. This could involve custom attention
        mechanisms or modifications to the standard scaled dot product attention calculation.

        Parameters: Mirror torch.nn.functional.scaled_dot_product_attention

        Returns:
        - torch.Tensor: The result of the attention computation.

        Raises:
        - NotImplementedError: If not implemented in a subclass.

        """
        raise NotImplementedError("This is an abstract base class")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """ "Defines the behavior of torch.nn.functional.scaled_dot_product_attention when the attn_bias is an AttnBias"""
        if kwargs is None:
            kwargs = {}
        if func != torch.nn.functional.scaled_dot_product_attention:
            raise NotImplementedError(
                "AttnBias only supports scaled_dot_product_attention"
            )
        if _any_need_materialization(args) or _any_need_materialization(kwargs):
            args = pytree.tree_map_only(
                AttnBias, lambda x: _materialize_if_needed(x), args
            )
            kwargs = pytree.tree_map_only(
                AttnBias, lambda x: _materialize_if_needed(x), kwargs
            )
            return func(*args, **kwargs)
        # Subclass is expected to have implemented a dispatch method
        return cls.dispatch(*args, **kwargs)


class TensorBias(AttnBias):
    """A subclass of AttnBias representing attention biases as a direct tensor.

    This class is used when the attention bias is already in a tensor format and needs to be directly
    used in calculations without any transformation or processing.

    """

    def __init__(self, bias: torch.Tensor):
        self.bias = bias

    def materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns the bias tensor.

        Since the bias is already a tensor, this method simply returns the bias without any modification.
        """
        return self.bias

    def needs_materialization(self) -> bool:
        """
        Always returns True for TensorBias.

        Indicates that the bias is already in tensor form and does not require any additional processing
        to be used in attention calculations.

        Returns:
        - bool: True, indicating the bias is already a tensor.
        """
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
        """
        Raises NotImplementedError.

        This method is not used for TensorBias, as the bias is already in tensor form and does not require
        any special dispatch logic for attention calculations. This method should never be called for TensorBias.

        Raises:
        - NotImplementedError: Always raised to indicate this method should not be called.
        """
        raise NotImplementedError(
            "TensorBias requires materialization, so this should never be called!"
        )

    def __repr__(self) -> str:
        return f"TensorBias(bias={self.bias})"


class CausalVariant(IntEnum):
    """
    Enum for causal variants used in attention mechanisms.

    Defines two types of causal biases:

    - UPPER_LEFT: Represents upper-left triangular bias for standard causal attention.
      Example:
      ```
      [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [1, 1, 1, 0]]
      ```
    - LOWER_RIGHT: Represents lower-right triangular bias, typically used in specific attention scenarios.
      Example:
      ```
      [[1, 1, 0, 0],
       [1, 1, 1, 0],
       [1, 1, 1, 1]]
      ```
    """

    UPPER_LEFT = auto()
    LOWER_RIGHT = auto()


class CausalBias(AttnBias):
    """
    A bias representing causal attention patterns, derived from AttnBias.

    This class is used for defining causal (triangular) attention biases.
    """

    def __init__(self, variant: CausalVariant, seq_len_q: int, seq_len_kv: int):
        """Initializes the CausalBias with the specified variant and sequence lengths.

        Parameters:
        - variant: The type of causal bias.
        - seq_len_q: The sequence length of the query.
        - seq_len_kv: The sequence length of the key/value.
        """
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
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        if is_causal:
            raise ValueError("CausalBias should not be used with causal=True")

        if (
            attn_mask.seq_len_q == attn_mask.seq_len_kv
            or attn_mask.variant == CausalVariant.UPPER_LEFT
        ):
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
                needs_padding = query.size(-1) % 8 != 0
                og_head_size = query.size(-1)
                if needs_padding:
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
                return _postprocess_flash_output(out, og_head_size)
            if can_use_efficient_attention(sdpa_params):
                compute_log_sumexp = False
                if _input_requires_grad(query, key, value):
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
            raise ValueError(
                f"CausalBias.variant must be a CausalVariant type, but found: {attn_mask.variant}"
            )

    def __repr__(self) -> str:
        return f"CausalBias(variant={self.variant.name}, seq_len_q={self.seq_len_q}, seq_len_kv={self.seq_len_kv})"
