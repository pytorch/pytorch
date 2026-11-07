```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# torch.nn.attention.flex_attention

```{eval-rst}
.. currentmodule:: torch.nn.attention.flex_attention
```
```{eval-rst}
.. py:module:: torch.nn.attention.flex_attention
```
<!-- Sphinx otherwise expands every typing overload into a separate signature. -->
```{eval-rst}
.. autofunction:: flex_attention(query: Tensor, key: Tensor, value: Tensor, score_mod: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor] | None = None, block_mask: BlockMask | None = None, scale: float | None = None, enable_gqa: bool = False, return_lse: bool = False, kernel_options: FlexKernelOptions | None = None, *, return_aux: AuxRequest | None = None) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, AuxOutput]
```
```{eval-rst}
.. autoclass:: AuxOutput
```
```{eval-rst}
.. autoclass:: AuxRequest
```

## BlockMask Utilities

```{eval-rst}
.. autofunction:: create_block_mask
```
```{eval-rst}
.. autofunction:: create_mask
```
```{eval-rst}
.. autofunction:: and_masks
```
```{eval-rst}
.. autofunction:: or_masks
```
```{eval-rst}
.. autofunction:: noop_mask
```

## FlexKernelOptions

```{eval-rst}
.. autoclass:: FlexKernelOptions
    :members:
    :undoc-members:
```

### ROCm backward options

The Triton backward kernel automatically selects optimized profiles for qualified gfx950 inputs.
Automatic selection requires the default scale, an identity score modification, and no captures.
Query, key, value, output, gradient, and statistics tensors must be contiguous.
It also requires one backward configuration and no auxiliary log-sum-exp gradient.
The following shapes use automatic selection:

| Dtype | Head dimension | Mask | `(batch, Hq, Hkv, length)` |
| --- | --- | --- | --- |
| BF16 | 64 | Default dense | `(4, 64, 8, 4096)` |
| BF16 | 96 | Default dense | `(4, 21, 21, 2048/4096)`, `(2, 8, 2, 2048/4096)` |
| BF16 | 128 | Causal | `(4, 64, 8, 4096)` |
| BF16 | 192 | Default dense | `(4, 10, 10, 2048/4096)`, `(2, 8, 2, 2048)` |
| BF16 | 256 | Default dense | `(4, 32, 32, 4096)`, `(4, 32, 8, 4096)` |
| FP16 | 64 | Default dense | `(1, 8, 2, 4096)` |
| FP16 | 128 | Causal | `(4, 64, 8, 2048/4096)`, `(2, 64, 8, 2048/4096)`, `(1, 64, 8, 4096)` |
| FP32 | 128 | Causal | `(1/2/4, 64, 8, 2048/4096)` |

Causal profiles require 128-by-128 blocks and contiguous metadata broadcast across batch and heads.
Automatic causal profiles use indexed traversal unless the caller guarantees contiguous block indices.
The compiler does not infer that guarantee from causal mask syntax.
Unsupported inputs retain existing kernel selection.
Explicit ROCm backward controls or backward tile and compiler settings bypass automatic profile selection.
Forward settings do not disable automatic backward selection.
Set `bwd_ROCM_SEPARATE_DQ=False` to disable automatic selection.
Custom backward choices retain precedence over automatic split execution.

For example, disable automatic ROCm backward profile selection with:

```python
kernel_options = {"BACKEND": "TRITON", "bwd_ROCM_SEPARATE_DQ": False}
```

The compiler controls scheduling, dot-product decomposition, and backward phase configuration internally.

## BlockMask

<!-- Document as_tuple separately so Sphinx does not expand its typing overloads. -->
```{eval-rst}
.. autoclass:: BlockMask
    :members:
    :undoc-members:
    :exclude-members: as_tuple

.. automethod:: BlockMask.as_tuple(flatten: bool = True) -> tuple[Any, ...]
```
