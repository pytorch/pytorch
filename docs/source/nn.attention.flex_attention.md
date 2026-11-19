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

## BlockMask

<!-- Document as_tuple separately so Sphinx does not expand its typing overloads. -->
```{eval-rst}
.. autoclass:: BlockMask
    :members:
    :undoc-members:
    :exclude-members: as_tuple

.. automethod:: BlockMask.as_tuple(flatten: bool = True) -> tuple[Any, ...]
```
