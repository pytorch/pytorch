---
name: docstring
description: Write docstrings for PyTorch functions and methods following PyTorch conventions. Use when writing or updating docstrings in PyTorch code.
---

# PyTorch Docstring Writing Guide

This skill describes how to write docstrings for functions and methods in the PyTorch project, following the conventions in `torch/_tensor_docs.py` and `torch/nn/functional.py`.

## General Principles

- Use **raw strings** (`r"""..."""`) for all docstrings to avoid issues with LaTeX/math backslashes
- Follow **Sphinx/reStructuredText** (reST) format for documentation
- Be **concise but complete** - include all essential information
- Always include **examples** when possible
- Use **cross-references** to related functions/classes

## Docstring Structure

### 1. Function Signature (First Line)

Start with the function signature showing all parameters:

```python
r"""function_name(param1, param2, *, kwarg1=default1, kwarg2=default2) -> ReturnType
```

**Notes:**
- Include the function name
- Show positional and keyword-only arguments (use `*` separator)
- Include default values
- Show return type annotation
- This line should NOT end with a period

### 2. Brief Description

Provide a one-line description of what the function does:

```python
r"""conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 2D convolution over an input image composed of several input
planes.
```

### 3. Mathematical Formulas (if applicable)

Use Sphinx math directives for mathematical expressions:

```python
.. math::
    \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
```

Or inline math: `:math:\`x^2\``

### 4. Cross-References

Link to related classes and functions using Sphinx roles:

- `:class:\`~torch.nn.ModuleName\`` - Link to a class
- `:func:\`torch.function_name\`` - Link to a function
- `:meth:\`~Tensor.method_name\`` - Link to a method
- `:attr:\`attribute_name\`` - Reference an attribute
- The `~` prefix shows only the last component (e.g., `Conv2d` instead of `torch.nn.Conv2d`)

**Example:**
```python
See :class:`~torch.nn.Conv2d` for details and output shape.
```

### 5. Notes and Warnings

Use admonitions for important information:

```python
.. note::
    This function doesn't work directly with NLLLoss,
    which expects the Log to be computed between the Softmax and itself.
    Use log_softmax instead (it's faster and has better numerical properties).

.. warning::
    :func:`new_tensor` always copies :attr:`data`. If you have a Tensor
    ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
    or :func:`torch.Tensor.detach`.
```

### 6. Args Section

Document all parameters with type annotations and descriptions:

```python
Args:
    input (Tensor): input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    weight (Tensor): filters of shape :math:`(\text{out\_channels} , kH , kW)`
    bias (Tensor, optional): optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride (int or tuple): the stride of the convolving kernel. Can be a single number or a
      tuple `(sH, sW)`. Default: 1
```

**Formatting rules:**
- Parameter name in **lowercase**
- Type in parentheses: `(Type)`, `(Type, optional)` for optional parameters
- Description follows the type
- For optional parameters, include "Default: ``value``" at the end
- Use double backticks for inline code: ``` ``None`` ```
- Indent continuation lines by 2 spaces

### 7. Keyword Args Section (if applicable)

Sometimes keyword arguments are documented separately:

```python
Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
        Default: if None, same :class:`torch.dtype` as this tensor.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, same :class:`torch.device` as this tensor.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
```

### 8. Returns Section (if needed)

Document the return value:

```python
Returns:
    Tensor: Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
        If ``hard=True``, the returned samples will be one-hot, otherwise they will
        be probability distributions that sum to 1 across `dim`.
```

Or simply include it in the function signature line if obvious from context.

### 9. Examples Section

Always include examples when possible:

```python
Examples::

    >>> inputs = torch.randn(33, 16, 30)
    >>> filters = torch.randn(20, 16, 5)
    >>> F.conv1d(inputs, filters)

    >>> # With square kernels and equal stride
    >>> filters = torch.randn(8, 4, 3, 3)
    >>> inputs = torch.randn(1, 4, 5, 5)
    >>> F.conv2d(inputs, filters, padding=1)
```

**Formatting rules:**
- Use `Examples::` with double colon
- Use `>>>` prompt for Python code
- Include comments with `#` when helpful
- Show actual output when it helps understanding (indent without `>>>`)

### 10. External References

Link to papers or external documentation:

```python
.. _Link Name:
    https://arxiv.org/abs/1611.00712
```

Reference them in text: ```See `Link Name`_```

## Method Types

### Native Python Functions

For regular Python functions, use a standard docstring:

```python
def relu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    # implementation
```

### C-Bound Functions (using add_docstr)

For C-bound functions, use `_add_docstr`:

```python
conv1d = _add_docstr(
    torch.conv1d,
    r"""
conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

See :class:`~torch.nn.Conv1d` for details and output shape.

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , kW)`
    ...
""",
)
```

### In-Place Variants

For in-place operations (ending with `_`), reference the original:

```python
add_docstr_all(
    "abs_",
    r"""
abs_() -> Tensor

In-place version of :meth:`~Tensor.abs`
""",
)
```

### Alias Functions

For aliases, simply reference the original:

```python
add_docstr_all(
    "absolute",
    r"""
absolute() -> Tensor

Alias for :func:`abs`
""",
)
```

## Common Patterns

### Shape Documentation

Use LaTeX math notation for tensor shapes:

```python
:math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
```

### Reusable Argument Definitions

For commonly used arguments, define them once and reuse:

```python
common_args = parse_kwargs(
    """
    dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
        Default: if None, same as this tensor.
"""
)

# Then use with .format():
r"""
...

Keyword args:
    {dtype}
    {device}
""".format(**common_args)
```

### Template Insertion

Insert reproducibility notes or other common text:

```python
r"""
{tf32_note}

{cudnn_reproducibility_note}
""".format(**reproducibility_notes, **tf32_notes)
```

## Complete Example

Here's a complete example showing all elements:

```python
def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
) -> Tensor:
    r"""
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
        logits (Tensor): `[..., num_features]` unnormalized log probabilities
        tau (float): non-negative scalar temperature
        hard (bool): if ``True``, the returned samples will be discretized as one-hot vectors,
              but will be differentiated as if it is the soft sample in autograd. Default: ``False``
        dim (int): A dimension along which softmax will be computed. Default: -1

    Returns:
        Tensor: Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
            If ``hard=True``, the returned samples will be one-hot, otherwise they will
            be probability distributions that sum to 1 across `dim`.

    .. note::
        This function is here for legacy reasons, may be removed from nn.Functional in the future.

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    """
    # implementation
```

## Quick Checklist

When writing a PyTorch docstring, ensure:

- [ ] Use raw string (`r"""`)
- [ ] Include function signature on first line
- [ ] Provide brief description
- [ ] Document all parameters in Args section with types
- [ ] Include default values for optional parameters
- [ ] Use Sphinx cross-references (`:func:`, `:class:`, `:meth:`)
- [ ] Add mathematical formulas if applicable
- [ ] Include at least one example in Examples section
- [ ] Add warnings/notes for important caveats
- [ ] Link to related module class with `:class:`
- [ ] Use proper math notation for tensor shapes
- [ ] Follow consistent formatting and indentation

## Common Sphinx Roles Reference

- `:class:\`~torch.nn.Module\`` - Class reference
- `:func:\`torch.function\`` - Function reference
- `:meth:\`~Tensor.method\`` - Method reference
- `:attr:\`attribute\`` - Attribute reference
- `:math:\`equation\`` - Inline math
- `:ref:\`label\`` - Internal reference
- ``` ``code`` ``` - Inline code (use double backticks)

## Additional Notes

- **Indentation**: Use 4 spaces for code, 2 spaces for continuation of parameter descriptions
- **Line length**: Try to keep lines under 100 characters when possible
- **Periods**: End sentences with periods, but not the signature line
- **Backticks**: Use double backticks for code: ``` ``True`` ``None`` ``False`` ```
- **Types**: Common types are `Tensor`, `int`, `float`, `bool`, `str`, `tuple`, `list`, etc.
