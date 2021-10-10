# The Tensor classes are added to this module by python_tensor.cpp
from typing import Optional, Tuple, List, Union

import torch
from torch import Tensor

# A workaround to support both TorchScript and MyPy:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
    DimOrDims = Optional[Union[int, Tuple[int], List[int]]]
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int
    DimOrDims = Optional[Tuple[int]]


__all__ = []

# All masked reduction/normalization operations have the same
# signatures. Here we introduce docstring templates that are applied
# to docstrings of reduction/normalization functions via
# _apply_docstring_templates decorator.

def _apply_docstring_templates(func):
    """Decorator that applies docstring templates to function docstring
    and returns the function instance.
    """

    docstring_templates = dict(
        masked_reduction_signature='''\
{function_name}(input, dim, keepdim=False, dtype=None, mask=None) -> Tensor''',
        masked_reduction_descr='''\
Returns {operation name} of :attr:`input` tensor along given dimension
:attr:`dim` while the :attr:`input` elements are masked out according
to the boolean tensor :attr:`mask`.''',
        masked_reduction_args='''\
If :attr:`keepdim` is ``True``, the output tensor is of the same
size as :attr:`input` except in the dimension(s) :attr:`dim` where
it is of size 1.  Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1
fewer dimension.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in {function name} computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored, the corresponding element of the output
tensor will have undefined value: it may or may not correspond to the
identity value of {operation name} operation; the choice may
correspond to the value that leads to the most efficient storage of
:attr:`output` tensor.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>`.

Args:
    input (Tensor): the input tensor
    dim (int): a dimension along which {function name} will be
      computed.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: ``torch.ones(input.shape, dtype=torch.bool)``.''',
        masked_reduction_example='''\
Example::

    >>> input = torch.tensor([[-3, -2, -1], [0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = torch.tensor([[True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])''',
        masked_reduction_identity='''\
The identity value of {function name}, which is used to start the reduction, is {identity_int32}.''',
        masked_reduction_identity_dtype='''\
The identity value of {function name}, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are {identity_float32}, {identity_uint8}, and {identity_int32}, respecitively.''',
        masked_normalization_args='''\
    TBD
''')

    # Apply function name info to docstring templates:
    templates = dict(
        (k, v.format_map(
            {'function_name': func.__name__,
             'function name': func.__name__.replace('_', ' '),
             'operation_name': '_'.join(func.__name__.split('_')[1:]),
             'operation name': ' '.join(func.__name__.split('_')[1:]),
             'identity_uint8': dict(masked_sum='0',
                                    masked_prod='1',
                                    masked_amax=0,
                                    masked_amin=2 ** 8 - 1).get(func.__name__),
             'identity_int32': dict(masked_sum='0',
                                    masked_prod='1',
                                    masked_amax=-2 ** 31,
                                    masked_amin=2 ** 31 - 1).get(func.__name__),
             'identity_float32': dict(masked_sum='0.0',
                                      masked_prod='1.0',
                                      masked_amax='-inf',
                                      masked_amin='inf').get(func.__name__)}
        )) for k, v in docstring_templates.items())

    # Apply docstring templates to function doctring:
    func.__doc__ = func.__doc__.format_map(templates)

    # Expose function as public symbol
    __all__.append(func.__name__)

    return func


def _reduction_identity(op_name: str, input: Tensor):
    """Return identity value as scalar tensor of a reduction operation on
    given input.
    """
    dtype: DType = input.dtype
    device = input.device
    if op_name == '_masked.sum':
        return torch.tensor(0, dtype=dtype, device=device)
    elif op_name == '_masked.prod':
        return torch.tensor(1, dtype=dtype, device=device)
    elif op_name == '_masked.amax':
        if torch.is_floating_point(input):
            return torch.tensor(-torch.inf, dtype=dtype, device=device)
        elif torch.is_signed(input):
            return torch.tensor(torch.iinfo(dtype).min, dtype=dtype, device=device)
        elif dtype == torch.uint8:
            return torch.tensor(0, dtype=dtype, device=device)
    elif op_name == '_masked.amin':
        if torch.is_floating_point(input):
            return torch.tensor(torch.inf, dtype=dtype, device=device)
        elif torch.is_signed(input):
            return torch.tensor(torch.iinfo(dtype).max, dtype=dtype, device=device)
        elif dtype == torch.uint8:
            return torch.tensor(255, dtype=dtype, device=device)
    raise NotImplementedError(f'identity of {op_name} on {dtype} input')


def _canonical_dim(dim: DimOrDims, ndim: int) -> Tuple[int, ...]:
    """Return dim argument as a tuple of sorted dim values.
    """
    dims: List[int] = []
    if dim is None:
        return tuple(range(ndim))
    ndim = max(ndim, 1)
    dim_ = (dim,) if isinstance(dim, int) else dim
    for d in dim_:
        if d in dims:
            raise RuntimeError(f'dim={d} appears multiple times in the list of dims')
        if d >= ndim or d < -ndim:
            raise IndexError(f'Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {d})')
        dims.append(d % ndim)
    return tuple(sorted(dims))


@_apply_docstring_templates
def sum(input: Tensor,
        dim: DimOrDims = None,
        *,
        keepdim: Optional[bool] = False,
        dtype: Optional[DType] = None,
        mask: Optional[Tensor] = None) -> Tensor:
    """
{masked_reduction_signature}

{masked_reduction_descr}

{masked_reduction_identity}

{masked_reduction_args}

{masked_reduction_example}
    >>> torch._masked.sum(input, 1, mask=mask)
    tensor([-4,  0])
    """
    if dtype is None:
        dtype = input.dtype
    # TODO: What follows is a reference implementation of a masked sum
    # operation that is to be replaced with an optimized one and
    # extended to support other layouts.
    if input.layout == torch.strided:
        mask_input = input if mask is None else torch.where(mask, input, torch.zeros_like(input))
        if dim is None:
            return torch.sum(mask_input, dtype=dtype)
        else:
            return torch.sum(mask_input, dim, bool(keepdim), dtype=dtype)
    else:
        raise NotImplementedError(f'masked_sum of {input.layout} tensor')


@_apply_docstring_templates
def prod(input: Tensor,
         dim: DimOrDims = None,
         *,
         keepdim: Optional[bool] = False,
         dtype: Optional[DType] = None,
         mask: Optional[Tensor] = None) -> Tensor:
    """
{masked_reduction_signature}

{masked_reduction_descr}

{masked_reduction_identity}

{masked_reduction_args}

{masked_reduction_example}
    >>> torch._masked.prod(input, 1, mask=mask)
    tensor([3,  1])
    """
    if input.layout == torch.strided:
        mask_input = input if mask is None else torch.where(mask, input, torch.ones_like(input))
        if dim is None:
            result = torch.prod(mask_input)
            if keepdim:
                result = result.reshape((1,) * mask_input.ndim)
        elif isinstance(dim, int):
            result = torch.prod(mask_input, dim, bool(keepdim), dtype=dtype)
        else:
            # Workaround https://github.com/pytorch/pytorch/issues/56586
            result = mask_input
            for d in reversed(_canonical_dim(dim, mask_input.ndim)):
                result = result.prod(dim=d, keepdim=bool(keepdim))
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result
    else:
        raise NotImplementedError(f'_masked.prod of {input.layout} tensor')


@_apply_docstring_templates
def amax(input: Tensor,
         dim: DimOrDims = None,
         *,
         keepdim: Optional[bool] = False,
         dtype: Optional[DType] = None,
         mask: Optional[Tensor] = None) -> Tensor:
    """
{masked_reduction_signature}

{masked_reduction_descr}

{masked_reduction_identity_dtype}

{masked_reduction_args}

{masked_reduction_example}
    >>> torch._masked.amax(input, 1, mask=mask)
    tensor([                  -1, -9223372036854775808])
    """
    if dtype is None:
        dtype = input.dtype
    if input.layout == torch.strided:
        if mask is None:
            mask_input = input
        else:
            identity = torch.empty_like(input)
            identity.fill_(_reduction_identity('_masked.amax', input))
            mask_input = torch.where(mask, input, identity)
        dim_ = _canonical_dim(dim, mask_input.ndim)
        return torch.amax(mask_input, dim_, bool(keepdim)).to(dtype=dtype)
    else:
        raise NotImplementedError(f'masked_amax of {input.layout} tensor')


@_apply_docstring_templates
def amin(input: Tensor,
         dim: DimOrDims = None,
         *,
         keepdim: Optional[bool] = False,
         dtype: Optional[DType] = None,
         mask: Optional[Tensor] = None) -> Tensor:
    """
{masked_reduction_signature}

{masked_reduction_descr}

{masked_reduction_identity_dtype}

{masked_reduction_args}

{masked_reduction_example}
    >>> torch._masked.amin(input, 1, mask=mask)
    tensor([                 -3, 9223372036854775807])
    """
    if dtype is None:
        dtype = input.dtype
    if input.layout == torch.strided:
        if mask is None:
            mask_input = input
        else:
            identity = torch.empty_like(input)
            identity.fill_(_reduction_identity('_masked.amin', input))
            mask_input = torch.where(mask, input, identity)
        dim_ = _canonical_dim(dim, mask_input.ndim)
        return torch.amin(mask_input, dim_, bool(keepdim)).to(dtype=dtype)
    else:
        raise NotImplementedError(f'_masked.amin of {input.layout} tensor')


def _output_mask(input: Tensor,
                 dim: DimOrDims = None,
                 *,
                 keepdim: Optional[bool] = False,
                 dtype: Optional[DType] = None,  # unused
                 mask: Optional[Tensor] = None) -> Tensor:
    """Return the output mask of an masked reduction operation.

    This function is equivalent to torch.any with broadcasting mask to
    input shape and supporting multiple dims. Used to implement masked
    equality check required in testing masked reductions.
    """
    if mask is None:
        if input.layout == torch.strided:
            outmask = torch.ones(input.shape, dtype=torch.bool, device=input.device)
        else:
            raise NotImplementedError(f'mask from layout {input.layout}')
    elif mask.ndim < input.ndim:
        outmask = torch.broadcast_to(mask.clone(), input.shape).to(dtype=torch.bool)
    elif mask.ndim > input.ndim:
        raise NotImplementedError("mask dimensionality higher than of input")
    else:
        outmask = mask.to(dtype=torch.bool)
    if isinstance(dim, tuple):
        # Workaround https://github.com/pytorch/pytorch/issues/56586
        for d in reversed(_canonical_dim(dim, input.ndim)):
            outmask = outmask.any(dim=d, keepdim=bool(keepdim))
    elif isinstance(dim, int):
        outmask = outmask.any(dim=dim, keepdim=bool(keepdim))
    else:
        assert dim is None
        outmask = outmask.any()
    return outmask
