.. currentmodule:: torch

.. _name_inference_reference-doc:

Named Tensors operator coverage
===============================

Please read :ref:`named_tensors-doc` first for an introduction to named tensors.

This document is a reference for *name inference*, a process that defines how
named tensors:

1. use names to provide additional automatic runtime correctness checks
2. propagate names from input tensors to output tensors

Below is a list of all operations that are supported with named tensors
and their associated name inference rules.

If you don't see an operation listed here, but it would help your use case, please
`search if an issue has already been filed <https://github.com/pytorch/pytorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22module%3A+named+tensor%22>`_ and if not, `file one <https://github.com/pytorch/pytorch/issues/new/choose>`_.

.. warning::
    The named tensor API is experimental and subject to change.

.. csv-table:: Supported Operations
   :header: API, Name inference rule
   :widths: 20, 20

   ":meth:`Tensor.abs`, :func:`torch.abs`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.abs_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.acos`, :func:`torch.acos`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.acos_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.add`, :func:`torch.add`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.add_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.addmm`, :func:`torch.addmm`",:ref:`contracts_away_dims-doc`
   :meth:`Tensor.addmm_`,:ref:`contracts_away_dims-doc`
   ":meth:`Tensor.addmv`, :func:`torch.addmv`",:ref:`contracts_away_dims-doc`
   :meth:`Tensor.addmv_`,:ref:`contracts_away_dims-doc`
   :meth:`Tensor.align_as`,See documentation
   :meth:`Tensor.align_to`,See documentation
   ":meth:`Tensor.all`, :func:`torch.all`",None
   ":meth:`Tensor.any`, :func:`torch.any`",None
   ":meth:`Tensor.asin`, :func:`torch.asin`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.asin_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.atan`, :func:`torch.atan`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.atan2`, :func:`torch.atan2`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.atan2_`,:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.atan_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.bernoulli`, :func:`torch.bernoulli`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.bernoulli_`,None
   :meth:`Tensor.bfloat16`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.bitwise_not`, :func:`torch.bitwise_not`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.bitwise_not_`,None
   ":meth:`Tensor.bmm`, :func:`torch.bmm`",:ref:`contracts_away_dims-doc`
   :meth:`Tensor.bool`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.byte`,:ref:`keeps_input_names-doc`
   :func:`torch.cat`,:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.cauchy_`,None
   ":meth:`Tensor.ceil`, :func:`torch.ceil`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.ceil_`,None
   :meth:`Tensor.char`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.chunk`, :func:`torch.chunk`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.clamp`, :func:`torch.clamp`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.clamp_`,None
   :meth:`Tensor.copy_`,:ref:`out_function_semantics-doc`
   ":meth:`Tensor.cos`, :func:`torch.cos`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.cos_`,None
   ":meth:`Tensor.cosh`, :func:`torch.cosh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.cosh_`,None
   ":meth:`Tensor.acosh`, :func:`torch.acosh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.acosh_`,None
   :meth:`Tensor.cpu`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.cuda`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.cumprod`, :func:`torch.cumprod`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.cumsum`, :func:`torch.cumsum`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.data_ptr`,None
   ":meth:`Tensor.deg2rad`, :func:`torch.deg2rad`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.deg2rad_`,None
   ":meth:`Tensor.detach`, :func:`torch.detach`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.detach_`,None
   ":attr:`Tensor.device`, :func:`torch.device`",None
   ":meth:`Tensor.digamma`, :func:`torch.digamma`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.digamma_`,None
   :meth:`Tensor.dim`,None
   ":meth:`Tensor.div`, :func:`torch.div`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.div_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.dot`, :func:`torch.dot`",None
   :meth:`Tensor.double`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.element_size`,None
   :func:`torch.empty`,:ref:`factory-doc`
   :func:`torch.empty_like`,:ref:`factory-doc`
   ":meth:`Tensor.eq`, :func:`torch.eq`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.erf`, :func:`torch.erf`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.erf_`,None
   ":meth:`Tensor.erfc`, :func:`torch.erfc`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.erfc_`,None
   ":meth:`Tensor.erfinv`, :func:`torch.erfinv`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.erfinv_`,None
   ":meth:`Tensor.exp`, :func:`torch.exp`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.exp_`,None
   :meth:`Tensor.expand`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.expm1`, :func:`torch.expm1`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.expm1_`,None
   :meth:`Tensor.exponential_`,None
   :meth:`Tensor.fill_`,None
   ":meth:`Tensor.flatten`, :func:`torch.flatten`",See documentation
   :meth:`Tensor.float`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.floor`, :func:`torch.floor`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.floor_`,None
   ":meth:`Tensor.frac`, :func:`torch.frac`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.frac_`,None
   ":meth:`Tensor.ge`, :func:`torch.ge`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.get_device`, :func:`torch.get_device`",None
   :attr:`Tensor.grad`,None
   ":meth:`Tensor.gt`, :func:`torch.gt`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.half`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.has_names`,See documentation
   ":meth:`Tensor.index_fill`, :func:`torch.index_fill`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.index_fill_`,None
   :meth:`Tensor.int`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.is_contiguous`,None
   :attr:`Tensor.is_cuda`,None
   ":meth:`Tensor.is_floating_point`, :func:`torch.is_floating_point`",None
   :attr:`Tensor.is_leaf`,None
   :meth:`Tensor.is_pinned`,None
   :meth:`Tensor.is_shared`,None
   ":meth:`Tensor.is_signed`, :func:`torch.is_signed`",None
   :attr:`Tensor.is_sparse`,None
   :attr:`Tensor.is_sparse_csr`,None
   :func:`torch.is_tensor`,None
   :meth:`Tensor.item`,None
   :attr:`Tensor.itemsize`,None
   ":meth:`Tensor.kthvalue`, :func:`torch.kthvalue`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.le`, :func:`torch.le`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.log`, :func:`torch.log`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.log10`, :func:`torch.log10`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.log10_`,None
   ":meth:`Tensor.log1p`, :func:`torch.log1p`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.log1p_`,None
   ":meth:`Tensor.log2`, :func:`torch.log2`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.log2_`,None
   :meth:`Tensor.log_`,None
   :meth:`Tensor.log_normal_`,None
   ":meth:`Tensor.logical_not`, :func:`torch.logical_not`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.logical_not_`,None
   ":meth:`Tensor.logsumexp`, :func:`torch.logsumexp`",:ref:`removes_dimensions-doc`
   :meth:`Tensor.long`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.lt`, :func:`torch.lt`",:ref:`unifies_names_from_inputs-doc`
   :func:`torch.manual_seed`,None
   ":meth:`Tensor.masked_fill`, :func:`torch.masked_fill`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.masked_fill_`,None
   ":meth:`Tensor.masked_select`, :func:`torch.masked_select`",Aligns mask up to input and then unifies_names_from_input_tensors
   ":meth:`Tensor.matmul`, :func:`torch.matmul`",:ref:`contracts_away_dims-doc`
   ":meth:`Tensor.mean`, :func:`torch.mean`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.median`, :func:`torch.median`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.nanmedian`, :func:`torch.nanmedian`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.mm`, :func:`torch.mm`",:ref:`contracts_away_dims-doc`
   ":meth:`Tensor.mode`, :func:`torch.mode`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.mul`, :func:`torch.mul`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.mul_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.mv`, :func:`torch.mv`",:ref:`contracts_away_dims-doc`
   :attr:`Tensor.names`,See documentation
   ":meth:`Tensor.narrow`, :func:`torch.narrow`",:ref:`keeps_input_names-doc`
   :attr:`Tensor.nbytes`,None
   :attr:`Tensor.ndim`,None
   :meth:`Tensor.ndimension`,None
   ":meth:`Tensor.ne`, :func:`torch.ne`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.neg`, :func:`torch.neg`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.neg_`,None
   :func:`torch.normal`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.normal_`,None
   ":meth:`Tensor.numel`, :func:`torch.numel`",None
   :func:`torch.ones`,:ref:`factory-doc`
   ":meth:`Tensor.pow`, :func:`torch.pow`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.pow_`,None
   ":meth:`Tensor.prod`, :func:`torch.prod`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.rad2deg`, :func:`torch.rad2deg`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.rad2deg_`,None
   :func:`torch.rand`,:ref:`factory-doc`
   :func:`torch.rand`,:ref:`factory-doc`
   :func:`torch.randn`,:ref:`factory-doc`
   :func:`torch.randn`,:ref:`factory-doc`
   :meth:`Tensor.random_`,None
   ":meth:`Tensor.reciprocal`, :func:`torch.reciprocal`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.reciprocal_`,None
   :meth:`Tensor.refine_names`,See documentation
   :meth:`Tensor.register_hook`,None
   :meth:`Tensor.rename`,See documentation
   :meth:`Tensor.rename_`,See documentation
   :attr:`Tensor.requires_grad`,None
   :meth:`Tensor.requires_grad_`,None
   :meth:`Tensor.resize_`,Only allow resizes that do not change shape
   :meth:`Tensor.resize_as_`,Only allow resizes that do not change shape
   ":meth:`Tensor.round`, :func:`torch.round`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.round_`,None
   ":meth:`Tensor.rsqrt`, :func:`torch.rsqrt`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.rsqrt_`,None
   ":meth:`Tensor.select`, :func:`torch.select`",:ref:`removes_dimensions-doc`
   :meth:`Tensor.short`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.sigmoid`, :func:`torch.sigmoid`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sigmoid_`,None
   ":meth:`Tensor.sign`, :func:`torch.sign`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sign_`,None
   ":meth:`Tensor.sgn`, :func:`torch.sgn`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sgn_`,None
   ":meth:`Tensor.sin`, :func:`torch.sin`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sin_`,None
   ":meth:`Tensor.sinh`, :func:`torch.sinh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sinh_`,None
   ":meth:`Tensor.asinh`, :func:`torch.asinh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.asinh_`,None
   :meth:`Tensor.size`,None
   ":meth:`Tensor.softmax`, :func:`torch.softmax`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.split`, :func:`torch.split`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.sqrt`, :func:`torch.sqrt`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sqrt_`,None
   ":meth:`Tensor.squeeze`, :func:`torch.squeeze`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.std`, :func:`torch.std`",:ref:`removes_dimensions-doc`
   :func:`torch.std_mean`,:ref:`removes_dimensions-doc`
   :meth:`Tensor.stride`,None
   ":meth:`Tensor.sub`, :func:`torch.sub`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.sub_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.sum`, :func:`torch.sum`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.tan`, :func:`torch.tan`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.tan_`,None
   ":meth:`Tensor.tanh`, :func:`torch.tanh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.tanh_`,None
   ":meth:`Tensor.atanh`, :func:`torch.atanh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.atanh_`,None
   :func:`torch.tensor`,:ref:`factory-doc`
   :meth:`Tensor.to`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.topk`, :func:`torch.topk`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.transpose`, :func:`torch.transpose`",:ref:`permutes_dimensions-doc`
   ":meth:`Tensor.trunc`, :func:`torch.trunc`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.trunc_`,None
   :meth:`Tensor.type`,None
   :meth:`Tensor.type_as`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.unbind`, :func:`torch.unbind`",:ref:`removes_dimensions-doc`
   :meth:`Tensor.unflatten`,See documentation
   :meth:`Tensor.uniform_`,None
   ":meth:`Tensor.var`, :func:`torch.var`",:ref:`removes_dimensions-doc`
   :func:`torch.var_mean`,:ref:`removes_dimensions-doc`
   :meth:`Tensor.zero_`,None
   :func:`torch.zeros`,:ref:`factory-doc`


.. _keeps_input_names-doc:

Keeps input names
^^^^^^^^^^^^^^^^^

All pointwise unary functions follow this rule as well as some other unary functions.

- Check names: None
- Propagate names: input tensor's names are propagated to the output.

::

    >>> x = torch.randn(3, 3, names=('N', 'C'))
    >>> x.abs().names
    ('N', 'C')

.. _removes_dimensions-doc:

Removes dimensions
^^^^^^^^^^^^^^^^^^

All reduction ops like :meth:`~Tensor.sum` remove dimensions by reducing
over the desired dimensions. Other operations like :meth:`~Tensor.select` and
:meth:`~Tensor.squeeze` remove dimensions.

Wherever one can pass an integer dimension index to an operator, one can also pass
a dimension name. Functions that take lists of dimension indices can also take in a
list of dimension names.

- Check names: If :attr:`dim` or :attr:`dims` is passed in as a list of names,
  check that those names exist in :attr:`self`.
- Propagate names: If the dimensions of the input tensor specified by :attr:`dim`
  or :attr:`dims` are not present in the output tensor, then the corresponding names
  of those dimensions do not appear in ``output.names``.

::

    >>> x = torch.randn(1, 3, 3, 3, names=('N', 'C', 'H', 'W'))
    >>> x.squeeze('N').names
    ('C', 'H', 'W')

    >>> x = torch.randn(3, 3, 3, 3, names=('N', 'C', 'H', 'W'))
    >>> x.sum(['N', 'C']).names
    ('H', 'W')

    # Reduction ops with keepdim=True don't actually remove dimensions.
    >>> x = torch.randn(3, 3, 3, 3, names=('N', 'C', 'H', 'W'))
    >>> x.sum(['N', 'C'], keepdim=True).names
    ('N', 'C', 'H', 'W')


.. _unifies_names_from_inputs-doc:

Unifies names from inputs
^^^^^^^^^^^^^^^^^^^^^^^^^

All binary arithmetic ops follow this rule. Operations that broadcast still
broadcast positionally from the right to preserve compatibility with unnamed
tensors. To perform explicit broadcasting by names, use :meth:`Tensor.align_as`.

- Check names: All names must match positionally from the right. i.e., in
  ``tensor + other``, ``match(tensor.names[i], other.names[i])`` must be true for all
  ``i`` in ``(-min(tensor.dim(), other.dim()) + 1, -1]``.
- Check names: Furthermore, all named dimensions must be aligned from the right.
  During matching, if we match a named dimension ``A`` with an unnamed dimension
  ``None``, then ``A`` must not appear in the tensor with the unnamed dimension.
- Propagate names: unify pairs of names from the right from both tensors to
  produce output names.

For example,

::

    # tensor: Tensor[   N, None]
    # other:  Tensor[None,    C]
    >>> tensor = torch.randn(3, 3, names=('N', None))
    >>> other = torch.randn(3, 3, names=(None, 'C'))
    >>> (tensor + other).names
    ('N', 'C')

Check names:

- ``match(tensor.names[-1], other.names[-1])`` is ``True``
- ``match(tensor.names[-2], tensor.names[-2])`` is ``True``
- Because we matched ``None`` in :attr:`tensor` with ``'C'``,
  check to make sure ``'C'`` doesn't exist in :attr:`tensor` (it does not).
- Check to make sure ``'N'`` doesn't exists in :attr:`other` (it does not).

Finally, the output names are computed with
``[unify('N', None), unify(None, 'C')] = ['N', 'C']``

More examples::

    # Dimensions don't match from the right:
    # tensor: Tensor[N, C]
    # other:  Tensor[   N]
    >>> tensor = torch.randn(3, 3, names=('N', 'C'))
    >>> other = torch.randn(3, names=('N',))
    >>> (tensor + other).names
    RuntimeError: Error when attempting to broadcast dims ['N', 'C'] and dims
    ['N']: dim 'C' and dim 'N' are at the same position from the right but do
    not match.

    # Dimensions aren't aligned when matching tensor.names[-1] and other.names[-1]:
    # tensor: Tensor[N, None]
    # other:  Tensor[      N]
    >>> tensor = torch.randn(3, 3, names=('N', None))
    >>> other = torch.randn(3, names=('N',))
    >>> (tensor + other).names
    RuntimeError: Misaligned dims when attempting to broadcast dims ['N'] and
    dims ['N', None]: dim 'N' appears in a different position from the right
    across both lists.

.. note::

    In both of the last examples, it is possible to align the tensors by names
    and then perform the addition. Use :meth:`Tensor.align_as` to align
    tensors by name or :meth:`Tensor.align_to` to align tensors to a custom
    dimension ordering.

.. _permutes_dimensions-doc:

Permutes dimensions
^^^^^^^^^^^^^^^^^^^

Some operations, like :meth:`Tensor.t()`, permute the order of dimensions. Dimension names
are attached to individual dimensions so they get permuted as well.

If the operator takes in positional index :attr:`dim`, it is also able to take a dimension
name as :attr:`dim`.

- Check names: If :attr:`dim` is passed as a name, check that it exists in the tensor.
- Propagate names: Permute dimension names in the same way as the dimensions that are
  being permuted.

::

    >>> x = torch.randn(3, 3, names=('N', 'C'))
    >>> x.transpose('N', 'C').names
    ('C', 'N')

.. _contracts_away_dims-doc:

Contracts away dims
^^^^^^^^^^^^^^^^^^^

Matrix multiply functions follow some variant of this. Let's go through
:func:`torch.mm` first and then generalize the rule for batch matrix multiplication.

For ``torch.mm(tensor, other)``:

- Check names: None
- Propagate names: result names are ``(tensor.names[-2], other.names[-1])``.

::

    >>> x = torch.randn(3, 3, names=('N', 'D'))
    >>> y = torch.randn(3, 3, names=('in', 'out'))
    >>> x.mm(y).names
    ('N', 'out')

Inherently, a matrix multiplication performs a dot product over two dimensions,
collapsing them. When two tensors are matrix-multiplied, the contracted dimensions
disappear and do not show up in the output tensor.

:func:`torch.mv`, :func:`torch.dot` work in a similar way: name inference does not
check input names and removes the dimensions that are involved in the dot product:

::

    >>> x = torch.randn(3, 3, names=('N', 'D'))
    >>> y = torch.randn(3, names=('something',))
    >>> x.mv(y).names
    ('N',)

Now, let's take a look at ``torch.matmul(tensor, other)``. Assume that ``tensor.dim() >= 2``
and ``other.dim() >= 2``.

- Check names: Check that the batch dimensions of the inputs are aligned and broadcastable.
  See :ref:`unifies_names_from_inputs-doc` for what it means for the inputs to be aligned.
- Propagate names: result names are obtained by unifying the batch dimensions and removing
  the contracted dimensions:
  ``unify(tensor.names[:-2], other.names[:-2]) + (tensor.names[-2], other.names[-1])``.

Examples::

    # Batch matrix multiply of matrices Tensor['C', 'D'] and Tensor['E', 'F'].
    # 'A', 'B' are batch dimensions.
    >>> x = torch.randn(3, 3, 3, 3, names=('A', 'B', 'C', 'D'))
    >>> y = torch.randn(3, 3, 3, names=('B', 'E', 'F'))
    >>> torch.matmul(x, y).names
    ('A', 'B', 'C', 'F')


Finally, there are fused ``add`` versions of many matmul functions. i.e., :func:`addmm`
and :func:`addmv`. These are treated as composing name inference for i.e. :func:`mm` and
name inference for :func:`add`.

.. _factory-doc:

Factory functions
^^^^^^^^^^^^^^^^^


Factory functions now take a new :attr:`names` argument that associates a name
with each dimension.

::

    >>> torch.zeros(2, 3, names=('N', 'C'))
    tensor([[0., 0., 0.],
            [0., 0., 0.]], names=('N', 'C'))

.. _out_function_semantics-doc:

out function and in-place variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A tensor specified as an ``out=`` tensor has the following behavior:

- If it has no named dimensions, then the names computed from the operation
  get propagated to it.
- If it has any named dimensions, then the names computed from the operation
  must be exactly equal to the existing names. Otherwise, the operation errors.

All in-place methods modify inputs to have names equal to the computed names
from name inference. For example:

::

    >>> x = torch.randn(3, 3)
    >>> y = torch.randn(3, 3, names=('N', 'C'))
    >>> x.names
    (None, None)

    >>> x += y
    >>> x.names
    ('N', 'C')
