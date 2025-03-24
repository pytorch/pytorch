# mypy: allow-untyped-defs
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter

from .lazy import LazyModuleMixin
from .module import Module


__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
    "PartialLinear",
]


class Identity(Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


class Linear(Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# This class exists solely to avoid triggering an obscure error when scripting
# an improperly quantized attention layer. See this issue for details:
# https://github.com/pytorch/pytorch/issues/58969
# TODO: fail fast on quantization API usage error, then remove this class
# and replace uses of it with plain Linear
class NonDynamicallyQuantizableLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )


class Bilinear(Module):
    r"""Applies a bilinear transformation to the incoming data: :math:`y = x_1^T A x_2 + b`.

    Args:
        in1_features: size of each first input sample, must be > 0
        in2_features: size of each second input sample, must be > 0
        out_features: size of each output sample, must be > 0
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(*, H_\text{in1})` where :math:`H_\text{in1}=\text{in1\_features}` and
          :math:`*` means any number of additional dimensions including none. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(*, H_\text{in2})` where :math:`H_\text{in2}=\text{in2\_features}`.
        - Output: :math:`(*, H_\text{out})` where :math:`H_\text{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40])
    """

    __constants__ = ["in1_features", "in2_features", "out_features"]
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if in1_features <= 0:
            raise ValueError(f"in1_features must be > 0, but got {in1_features}")
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in1_features, in2_features), **factory_kwargs)
        )

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        return F.bilinear(input1, input2, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )


class LazyLinear(LazyModuleMixin, Linear):
    r"""A :class:`torch.nn.Linear` module where `in_features` is inferred.

    In this module, the `weight` and `bias` are of :class:`torch.nn.UninitializedParameter`
    class. They will be initialized after the first call to ``forward`` is done and the
    module will become a regular :class:`torch.nn.Linear` module. The ``in_features`` argument
    of the :class:`Linear` is inferred from the ``input.shape[-1]``.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`


    """

    cls_to_become = Linear  # type: ignore[assignment]
    weight: UninitializedParameter
    bias: UninitializedParameter  # type: ignore[assignment]

    def __init__(
        self, out_features: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # bias is hardcoded to False to avoid creating tensor
        # that will soon be overwritten.
        super().__init__(0, 0, False)
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_features = out_features
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize((self.out_features, input.shape[-1]))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()
        if self.in_features == 0:
            assert input.shape[-1] == self.weight.shape[-1], (
                f"The in_features inferred from input: {input.shape[-1]} "
                f"is not equal to in_features from self.weight: "
                f"{self.weight.shape[-1]}"
            )
            self.in_features = input.shape[-1]


class PartialLinear(Module):
    r"""Applies a linear transformation where each output feature connects to only the top-k
    input features by weight magnitude: :math:`y = x * (M \odot W^T) + b`,
    where $\odot$ is the element-wise product and $M$ is a binary mask.

    This module implements a form of structured sparsity that can reduce computation
    and memory usage during inference.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        top_k: number of weights to retain per output feature (default: in_features // 2)
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        update_mask_every: update the mask every N forward passes during training (default: 50)

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        mask: binary mask of shape :math:`(\text{out\_features}, \text{in\_features})`
            indicating which weights are retained (1) or pruned (0)
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
        is_sparse_forward: when True, uses a completely sparse computation for forward pass

    Examples::

        >>> m = nn.PartialLinear(20, 30, top_k=5)  # Each output connected to top 5 inputs
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ["in_features", "out_features", "top_k", "update_mask_every", "is_sparse_forward"]
    in_features: int
    out_features: int
    top_k: int
    update_mask_every: int
    is_sparse_forward: bool
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        top_k: int = None,
        bias: bool = True,
        update_mask_every: int = 50,
        is_sparse_forward: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Default to half of the input features if not specified
        if top_k is None:
            top_k = max(1, in_features // 2)
        
        if top_k <= 0 or top_k > in_features:
            raise ValueError(f"top_k must be between 1 and {in_features}, got {top_k}")
        
        self.top_k = top_k
        self.update_mask_every = update_mask_every
        self.is_sparse_forward = is_sparse_forward
        self._forward_counter = 0
        
        # Create a full weight matrix
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        # Create a binary mask for the weights
        mask = torch.ones((out_features, in_features), **factory_kwargs)
        self.register_buffer('mask', mask)
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard initialization as in nn.Linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        
        # Reset the mask to all ones
        self.mask.fill_(1.0)
        
        # Initialize the mask to keep only top-k weights
        self._update_mask()

    def _update_mask(self):
        with torch.no_grad():
            # Compute the magnitude of weights
            weight_mag = self.weight.abs()
            
            # Create a new binary mask
            new_mask = torch.zeros_like(self.mask)
            
            # For each output feature, find the top-k input connections
            _, top_k_indices = weight_mag.topk(self.top_k, dim=1)
            
            # Set mask to 1 for top-k weights for each output
            for i in range(self.out_features):
                new_mask[i, top_k_indices[i]] = 1.0
            
            # Update the mask
            self.mask.copy_(new_mask)

    def _sparse_forward(self, input):
        # Create output tensor
        output = torch.zeros(input.shape[:-1] + (self.out_features,), 
                          device=input.device, dtype=input.dtype)
        
        # Get non-zero indices of the mask
        nonzero_indices = self.mask.nonzero(as_tuple=True)
        
        # For each non-zero weight, add its contribution to the output
        for out_idx, in_idx in zip(*nonzero_indices):
            # Get the corresponding weight
            weight_val = self.weight[out_idx, in_idx]
            
            # Get the corresponding input values (handle batched inputs)
            if input.dim() > 1:
                in_vals = input[..., in_idx]
                # Add contribution to output (broadcasting handles batched inputs)
                output[..., out_idx] += in_vals * weight_val
            else:
                # For single input vector
                output[out_idx] += input[in_idx] * weight_val
        
        # Add bias if needed
        if self.bias is not None:
            output += self.bias
        
        return output

    def forward(self, input: Tensor) -> Tensor:
        # During training, periodically update the mask
        if self.training:
            self._forward_counter += 1
            if self._forward_counter >= self.update_mask_every:
                self._update_mask()
                self._forward_counter = 0
        
        # Use sparse computation if requested
        if self.is_sparse_forward:
            return self._sparse_forward(input)
        
        # Apply the mask to the weights
        masked_weight = self.weight * self.mask
        
        # Use the masked weights for the linear transformation
        return F.linear(input, masked_weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"top_k={self.top_k}, bias={self.bias is not None}, "
                f"update_mask_every={self.update_mask_every}, "
                f"is_sparse_forward={self.is_sparse_forward}")
