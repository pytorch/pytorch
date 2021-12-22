# -*- coding: utf-8 -*-

from typing import Optional, Tuple, List, Union, Any

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
        reduction_signature='''\
{function_name}(input, {operation_args}, *, {operation_kwargs}) -> Tensor''',
        reduction_descr='''\
Returns {operation name} of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.''',
        reduction_args='''\
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in {operation name} computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of {operation name} operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    {args_declarations}

Keyword args:
    {kwargs_declarations}''',
        reduction_example='''\
Example::

    >>> input = {example_input}
    >>> input
    {indent_example_input}
    >>> mask = {example_mask}
    >>> mask
    {indent_example_mask}
    >>> {full_function_name}(input, {example_args}, mask=mask)
    {indent_example_output}
''',
        reduction_identity='''\
The identity value of {operation name} operation, which is used to start the reduction, is ``{identity_int32}``.''',
        reduction_identity_dtype='''\
The identity value of {operation name} operation, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are ``{identity_float32}``, ``{identity_uint8}``, and ``{identity_int32}``, respectively.''',
        normalization_signature='''\
{function_name}(input, {operation_args}, *, {operation_kwargs}) -> Tensor''',
        normalization_descr='''\
Returns {operation name} of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

{definition}''',
        normalization_args='''\
The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
{operation name} computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the {operation name} output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    {args_declarations}

Keyword args:
    {kwargs_declarations}''',
        normalization_example='''\
Example::

    >>> input = {example_input}
    >>> input
    {indent_example_input}
    >>> mask = {example_mask}
    >>> mask
    {indent_example_mask}
    >>> {full_function_name}(input, {example_args}, mask=mask)
    {indent_example_output}
''')

    args_and_kwargs = dict(
        # argument name sufficies separated by double underscore will
        # be removed in the final documentation string.
        sum=(('dim',), ('keepdim=False', 'dtype=None', 'mask=None')),
        prod=(('dim',), ('keepdim=False', 'dtype=None', 'mask=None')),
        amin=(('dim',), ('keepdim=False', 'dtype=None', 'mask=None')),
        amax=(('dim',), ('keepdim=False', 'dtype=None', 'mask=None')),
        mean=(('dim',), ('keepdim=False', 'dtype=None', 'mask=None')),
        norm=(('ord', 'dim',), ('keepdim=False', 'dtype=None', 'mask=None')),
        var=(('dim', 'unbiased'), ('keepdim=False', 'dtype=None', 'mask=None')),
        softmax=(('dim__as_int',), ('dtype=None', 'mask=None')),
        log_softmax=(('dim__as_int',), ('dtype=None', 'mask=None')),
        softmin=(('dim__as_int',), ('dtype=None', 'mask=None')),
        normalize=(('ord__required', 'dim__as_int',), ('eps=1e-12', 'dtype=None', 'mask=None')),
    )

    argument_declarations = dict(
        dim='''\
dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
  Default: None that is equivalent to ``tuple(range(input.ndim))``.''',
        dim__as_int='''\
dim (int): the dimension along which {operation name} is computed.''',
        ord='''\
ord (int, float, optional): the order of vector norm. Default: 2.
  See :func:`torch.linalg.vector_norm` for a list of supported norms.''',
        ord__required='''\
ord (int, float): the order of vector norm. Default: 2.
  See :func:`torch.linalg.vector_norm` for a list of supported norms.''',
        unbiased='''\
unbiased (bool): when True, use Bessel’s correction, otherwise, compute
  the uncorrected sample variance.''',
        eps='''\
eps (float, optional): small value to avoid division by zero. Default: {default}.''',
        keepdim='''\
keepdim (bool, optional): whether the output tensor has
  :attr:`dim` retained or not. Default: {default}.''',
        dtype='''\
dtype (:class:`torch.dtype`, optional): the desired data type
  of returned tensor.  If specified, the input tensor is
  casted to :attr:`dtype` before the operation is
  performed. Default: {default}.''',
        mask='''\
mask (:class:`torch.Tensor`, optional): the boolean tensor
  containing the binary mask of validity of input tensor
  elements.
  Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.''')

    definitions = dict(
        softmax='''\
Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Softmax of i-th element in ``x`` is
defined as ``exp(x[i])/sum(exp(x))``.''',
        log_softmax='''\
Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. LogSoftmax of i-th element in ``x`` is
defined as ``log(exp(x[i])/sum(exp(x)))``.''',
        softmin='''\
Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Softmin of i-th element in ``x`` is
defined as ``exp(-x[i])/sum(exp(-x))``.''',
        normalize='''\
Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Normalize of i-th element in ``x`` is
defined as ``x[i]/max(norm(x, p), eps)``.''')

    reduction_names = dict(
        sum='sum',
        prod='product',
        amax='maximum',
        amin='minimum',
        mean='mean',
        norm='norm',
        var='variance')

    normalization_names = dict(
        softmax='softmax',
        log_softmax='log_softmax',
        softmin='softmin',
        normalize='normalize')

    operation_names = dict()
    operation_names.update(reduction_names)
    operation_names.update(normalization_names)

    # Default example data:
    example_dim = 1
    example_input = torch.tensor([[-3, -2, -1], [0, 1, 2]])
    example_mask = torch.tensor([[True, False, True], [False, False, False]])
    example_args: Tuple[Any, ...]
    if func.__name__ in {'norm', 'normalize'}:
        example_args = (2.0, example_dim)
        example_input = example_input.to(dtype=torch.float32)
    elif func.__name__ in {'var'}:
        example_args = (example_dim, False)
    else:
        example_args = (example_dim,)

    operation_args: Tuple[str, ...]
    operation_kwargs: Tuple[str, ...]
    operation_args, operation_kwargs = args_and_kwargs[func.__name__]
    arg_declarations = [
        '\n    '.join(argument_declarations
                      .get(a, f'{a.split("__", 1)[0]}: TBD.')
                      .splitlines())
        for a in operation_args]
    kwarg_declarations = [
        '\n    '.join(argument_declarations
                      .get(a.split('=', 1)[0], f'{a.split("__", 1)[0]}: TBD.')
                      .format(default=a.split('=', 1)[1])
                      .splitlines())
        for a in operation_kwargs]

    if func.__name__ in reduction_names:
        op_kind = 'reduction'
        doc_sections = ['signature', 'descr', 'identity', 'args', 'example']
    elif func.__name__ in normalization_names:
        op_kind = 'normalization'
        doc_sections = ['signature', 'descr', 'args', 'example']
        example_input = example_input.to(dtype=torch.float32)
    else:
        assert 0  # add function name to operation names dictionaries
    example_output = func(example_input, *example_args, mask=example_mask)

    template_data = {'function_name': func.__name__,
                     'full_function_name': func.__module__ + '.' + func.__name__,
                     'operation name': operation_names[func.__name__],
                     'operation_args': ', '.join(a.split('__', 1)[0] for a in operation_args),
                     'operation_kwargs': ', '.join(a.split('__', 1)[0] for a in operation_kwargs),
                     # one-line representation of a tensor:
                     'example_input': ' '.join(str(example_input).split()),
                     'example_args': ', '.join(map(str, example_args)),
                     'example_mask': ' '.join(str(example_mask).split()),
                     # multi-line representation of a tensor with indent
                     'indent_example_input': ('\n    ').join(str(example_input).splitlines()),
                     'indent_example_mask': ('\n    ').join(str(example_mask).splitlines()),
                     'indent_example_output': ('\n    ').join(str(example_output).splitlines())}

    if func.__name__ in reduction_names:
        template_data.update(
            identity_uint8=_reduction_identity(func.__name__, torch.tensor(0, dtype=torch.uint8)),
            identity_int32=_reduction_identity(func.__name__, torch.tensor(0, dtype=torch.int32)),
            identity_float32=_reduction_identity(func.__name__, torch.tensor(0, dtype=torch.float32)))
        if func.__name__ == 'norm':
            template_data.update(
                identity_ord_ninf=_reduction_identity(
                    func.__name__, torch.tensor(0, dtype=torch.float32), float('-inf')))
    elif func.__name__ in normalization_names:
        template_data.update(definition=definitions[func.__name__])
    else:
        assert 0  # add function name to operation names dictionaries
    template_data.update(args_declarations=('\n    '.join(arg_declarations)).format_map(template_data))
    template_data.update(kwargs_declarations=('\n    '.join(kwarg_declarations)).format_map(template_data))

    # Apply function name info to docstring templates:
    templates = dict((k, v.format_map(template_data))
                     for k, v in docstring_templates.items() if k.startswith(op_kind))
    templates.update((k, v.format_map(template_data) if isinstance(v, str) else v) for k, v in template_data.items())

    # Apply docstring templates to function doctring:
    if func.__doc__ is None:
        doc_template = '\n\n'.join([f'{{{op_kind}_{sec}}}' for sec in doc_sections])
    else:
        doc_template = func.__doc__
    func.__doc__ = doc_template.format_map(templates)

    # Expose function as public symbol
    __all__.append(func.__name__)

    return func


def _reduction_identity(op_name: str, input: Tensor, *args):
    """Return identity value as scalar tensor of a reduction operation on
    given input, or None, if the identity value cannot be uniquely
    defined for the given input.

    The identity value of the operation is defined as the initial
    value to reduction operation that has a property ``op(op_identity,
    value) == value`` for any value in the domain of the operation.
    Or put it another way, including or exlucing the identity value in
    a list of operands will not change the reduction result.

    See https://github.com/pytorch/rfcs/pull/27 for more information.

    """
    dtype: DType = input.dtype
    device = input.device
    op_name = op_name.rsplit('.', 1)[-1]  # lstrip module name when present
    if op_name == 'sum':
        return torch.tensor(0, dtype=dtype, device=device)
    elif op_name == 'prod':
        return torch.tensor(1, dtype=dtype, device=device)
    elif op_name == 'amax':
        if torch.is_floating_point(input):
            return torch.tensor(-torch.inf, dtype=dtype, device=device)
        elif torch.is_signed(input) or dtype == torch.uint8:
            return torch.tensor(torch.iinfo(dtype).min, dtype=dtype, device=device)
    elif op_name == 'amin':
        if torch.is_floating_point(input):
            return torch.tensor(torch.inf, dtype=dtype, device=device)
        elif torch.is_signed(input) or dtype == torch.uint8:
            return torch.tensor(torch.iinfo(dtype).max, dtype=dtype, device=device)
    elif op_name == 'mean':
        # Strictly speaking, the identity value of the mean operation
        # is the mean of the input. Since the mean value depends on
        # the dim argument and it may be a non-scalar tensor, we
        # consider the identity value of the mean operation ambiguous.
        # Moreover, the mean value of empty input is undefined.
        return None
    elif op_name == 'norm':
        ord = args[0] if args else 2
        if ord == float('-inf'):
            assert torch.is_floating_point(input), input.dtype
            return torch.tensor(torch.inf, dtype=dtype, device=device)
        return torch.tensor(0, dtype=dtype, device=device)
    elif op_name == 'var':
        return None
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


def _input_mask(input: Tensor, *args, **kwargs) -> Tensor:
    """Return canonical input mask.
    Canonical input mask is a boolean tensor with the same shape as
    input and with (broadcasted) content of mask, if specified.
    """
    mask = kwargs.get('mask')
    if mask is None:
        inmask = input.new_ones(input.shape, dtype=torch.bool)
    elif mask.ndim < input.ndim:
        inmask = torch.broadcast_to(mask.clone(), input.shape).to(dtype=torch.bool)
    elif mask.ndim > input.ndim:
        raise IndexError("_input_mask expected broadcastable mask (got mask dimensionality higher than of the input)")
    elif mask.shape != input.shape:
        inmask = torch.broadcast_to(mask.clone(), input.shape).to(dtype=torch.bool)
    else:
        inmask = mask.to(dtype=torch.bool)
    return inmask


def _output_mask(op, input: Tensor, *args, **kwargs) -> Tensor:
    """Return output mask of masked operation applied to given arguments.
    """
    if callable(op):
        is_reduction = op.__name__ in {'sum', 'prod', 'amax', 'amin', 'mean', 'norm', 'var'}
        is_normalization = op.__name__ in {'softmax', 'log_softmax', 'softmin', 'normalize'}
        if is_reduction:
            if op.__name__ == 'norm':
                if args:
                    args = args[1:]  # lstrip ord argument
            dim = args[0] if args else kwargs.get('dim')
            outmask = _input_mask(input, *args, **kwargs)
            keepdim = kwargs.get('keepdim', False)
            dim_ = _canonical_dim(dim, input.ndim)
            # Workaround https://github.com/pytorch/pytorch/issues/56586
            for d in reversed(dim_):
                outmask = outmask.any(dim=d, keepdim=bool(keepdim))
            return outmask
        elif is_normalization:
            return _input_mask(input, *args, **kwargs)
        else:
            raise ValueError(f'_output_mask expected masked operation (got callable {op.__module__}.{op.__name__})')
    else:
        raise ValueError(f'_output_mask expected masked operation (got {type(op).__name__} object)')


@_apply_docstring_templates
def sum(input: Tensor,
        dim: DimOrDims = None,
        *,
        keepdim: Optional[bool] = False,
        dtype: Optional[DType] = None,
        mask: Optional[Tensor] = None) -> Tensor:
    # __doc__ is generated by _apply_docstring_templates decorator
    if dtype is None:
        dtype = input.dtype
    # TODO: What follows is a reference implementation of a masked sum
    # operation that is to be replaced with an optimized one and
    # extended to support other layouts.
    if input.layout == torch.strided:
        mask_input = input if mask is None else torch.where(mask, input, input.new_zeros([]))
        dim_ = _canonical_dim(dim, input.ndim)
        return torch.sum(mask_input, dim_, bool(keepdim), dtype=dtype)
    else:
        raise ValueError(f'masked sum expects strided tensor (got {input.layout} tensor)')


@_apply_docstring_templates
def prod(input: Tensor,
         dim: DimOrDims = None,
         *,
         keepdim: Optional[bool] = False,
         dtype: Optional[DType] = None,
         mask: Optional[Tensor] = None) -> Tensor:
    # __doc__ is generated by _apply_docstring_templates decorator
    if input.layout == torch.strided:
        mask_input = input if mask is None else torch.where(mask, input, torch.ones_like(input))
        dim_ = _canonical_dim(dim, input.ndim)

        # Workaround https://github.com/pytorch/pytorch/issues/56586
        result = mask_input
        for d in reversed(dim_):
            result = result.prod(dim=d, keepdim=bool(keepdim))
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result
    else:
        raise ValueError(f'masked prod expects strided tensor (got {input.layout} tensor)')


@_apply_docstring_templates
def amax(input: Tensor,
         dim: DimOrDims = None,
         *,
         keepdim: Optional[bool] = False,
         dtype: Optional[DType] = None,
         mask: Optional[Tensor] = None) -> Tensor:
    """\
{reduction_signature}

{reduction_descr}

{reduction_identity_dtype}

{reduction_args}

{reduction_example}"""
    if dtype is None:
        dtype = input.dtype
    if input.layout == torch.strided:
        if mask is None:
            mask_input = input
        else:
            identity = input.new_full([], _reduction_identity('amax', input))
            mask_input = torch.where(mask, input, identity)
        dim_ = _canonical_dim(dim, mask_input.ndim)
        return torch.amax(mask_input, dim_, bool(keepdim)).to(dtype=dtype)
    else:
        raise ValueError(f'masked amax expects strided tensor (got {input.layout} tensor)')


@_apply_docstring_templates
def amin(input: Tensor,
         dim: DimOrDims = None,
         *,
         keepdim: Optional[bool] = False,
         dtype: Optional[DType] = None,
         mask: Optional[Tensor] = None) -> Tensor:
    """\
{reduction_signature}

{reduction_descr}

{reduction_identity_dtype}

{reduction_args}

{reduction_example}"""
    if dtype is None:
        dtype = input.dtype
    if input.layout == torch.strided:
        if mask is None:
            mask_input = input
        else:
            identity = input.new_full([], _reduction_identity('amin', input))
            mask_input = torch.where(mask, input, identity)
        dim_ = _canonical_dim(dim, mask_input.ndim)
        return torch.amin(mask_input, dim_, bool(keepdim)).to(dtype=dtype)
    else:
        raise ValueError(f'masked amin expects strided tensor (got {input.layout} tensor)')


@_apply_docstring_templates
def mean(input: Tensor,
         dim: DimOrDims = None,
         *,
         keepdim: Optional[bool] = False,
         dtype: Optional[DType] = None,
         mask: Optional[Tensor] = None) -> Tensor:
    """\
{reduction_signature}

{reduction_descr}

By definition, the identity value of a mean operation is the mean
value of the tensor. If all elements of the input tensor along given
dimension(s) :attr:`dim` are masked-out, the identity value of the
mean is undefined.  Due to this ambiguity, the elements of output
tensor with strided layout, that correspond to fully masked-out
elements, have ``nan`` values.

{reduction_args}

{reduction_example}"""
    if dtype is None:
        dtype = input.dtype
    if input.layout == torch.strided:
        inmask = _input_mask(input, mask=mask)
        count = sum(inmask.new_ones(input.shape, dtype=torch.int64), dim, keepdim=keepdim, mask=inmask)
        total = sum(input, dim, keepdim=keepdim, dtype=dtype, mask=inmask)
        return total / count
    else:
        raise ValueError(f'masked sum expects strided tensor (got {input.layout} tensor)')


@_apply_docstring_templates
def norm(input: Tensor,
         ord: Optional[float] = 2.0,
         dim: DimOrDims = None,
         *,
         keepdim: Optional[bool] = False,
         dtype: Optional[DType] = None,
         mask: Optional[Tensor] = None) -> Tensor:
    """\
{reduction_signature}

{reduction_descr}

The identity value of norm operation, which is used to start the
reduction, is ``{identity_float32}``, except for ``ord=-inf`` it is
``{identity_ord_ninf}``.

{reduction_args}

{reduction_example}"""
    if dtype is None:
        dtype = input.dtype
    if input.layout == torch.strided:
        identity = input.new_full([], _reduction_identity('norm', input, ord))
        mask_input = input if mask is None else torch.where(mask, input, identity)
        dim_ = _canonical_dim(dim, input.ndim)
        return torch.linalg.vector_norm(mask_input, ord, dim_, bool(keepdim), dtype=dtype)
    else:
        raise ValueError(f'masked norm expects strided tensor (got {input.layout} tensor)')


@_apply_docstring_templates
def var(input: Tensor,
        dim: DimOrDims = None,
        unbiased: Optional[bool] = False,
        *,
        keepdim: Optional[bool] = False,
        dtype: Optional[DType] = None,
        mask: Optional[Tensor] = None) -> Tensor:
    """\
{reduction_signature}

{reduction_descr}

The identity value of sample variance operation is undefined.  The
elements of output tensor with strided layout, that correspond to
fully masked-out elements, have ``nan`` values.

{reduction_args}

{reduction_example}"""
    if dtype is None:
        dtype = input.dtype
        if not (dtype.is_floating_point or dtype.is_complex):
            dtype = torch.float32
    compute_dtype = dtype
    if not (compute_dtype.is_floating_point or compute_dtype.is_complex):
        compute_dtype = torch.float32
    if input.layout == torch.strided:
        inmask = _input_mask(input, mask=mask)
        count = sum(inmask.new_ones(input.shape, dtype=torch.int64), dim, keepdim=True, mask=inmask)
        sample_total = sum(input, dim, keepdim=True, dtype=dtype, mask=inmask)
        # TODO: replace torch.subtract/divide/square/maximum with
        # masked subtract/divide/square/maximum when these will be
        # available.
        sample_mean = torch.divide(sample_total, count)
        x = torch.subtract(input, sample_mean)
        total = sum(x * x.conj(), dim, keepdim=keepdim, dtype=compute_dtype, mask=inmask)
        if not keepdim:
            count = count.reshape(total.shape)
        if unbiased:
            count = torch.subtract(count, 1)
            count = torch.maximum(count, count.new_zeros([]))
        return torch.divide(total, count).to(dtype=dtype)
    else:
        raise ValueError(f'masked var expects strided tensor (got {input.layout} tensor)')


@_apply_docstring_templates
def softmax(input: Tensor,
            dim: int,
            *,
            dtype: Optional[DType] = None,
            mask: Optional[Tensor] = None) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)[0]
    if input.layout == torch.strided:
        fill = input.new_full([], _reduction_identity('amax', input))
        inmask = _input_mask(input, mask=mask)
        mask_input = torch.where(inmask, input, fill)
        return torch.nn.functional.softmax(mask_input, dim_, dtype=dtype)
    else:
        raise ValueError(f'masked softmax expects strided tensor (got {input.layout} tensor)')


@_apply_docstring_templates
def log_softmax(input: Tensor,
                dim: int,
                *,
                dtype: Optional[DType] = None,
                mask: Optional[Tensor] = None) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)[0]
    if input.layout == torch.strided:
        fill = input.new_full([], _reduction_identity('amax', input))
        inmask = _input_mask(input, mask=mask)
        mask_input = torch.where(inmask, input, fill)
        return torch.nn.functional.log_softmax(mask_input, dim_, dtype=dtype)
    else:
        raise ValueError(f'masked log_softmax expects strided tensor (got {input.layout} tensor)')


@_apply_docstring_templates
def softmin(input: Tensor,
            dim: int,
            *,
            dtype: Optional[DType] = None,
            mask: Optional[Tensor] = None) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)[0]
    if input.layout == torch.strided:
        fill = input.new_full([], _reduction_identity('amin', input))
        inmask = _input_mask(input, mask=mask)
        mask_input = torch.where(inmask, input, fill)
        return torch.nn.functional.softmin(mask_input, dim_, dtype=dtype)
    else:
        raise ValueError(f'masked softmin expects strided tensor (got {input.layout} tensor)')


@_apply_docstring_templates
def normalize(input: Tensor,
              ord: float,
              dim: int,
              *,
              eps: float = 1e-12,
              dtype: Optional[DType] = None,
              mask: Optional[Tensor] = None) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)[0]
    if input.layout == torch.strided:
        nrm_ = norm(input, ord, dim, keepdim=True, dtype=dtype, mask=mask)
        # TODO: replace torch.maximum with masked maximum when available.
        denom = torch.maximum(nrm_, nrm_.new_full([], eps))
        # TODO: eliminate mask_input as unnecessary when using masked divide.
        inmask = _input_mask(input, mask=mask)
        mask_input = input if mask is None else torch.where(inmask, input, input.new_zeros([]))
        # TODO: replace torch.divide with masked divide when available.
        return torch.divide(mask_input, denom)
    else:
        raise ValueError(f'masked normalize expects strided tensor (got {input.layout} tensor)')
