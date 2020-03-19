.. currentmodule:: torch

.. _tensor-view-doc:

Tensor Views
=============

PyTorch allows a tensor to be a ``View`` of an existing tensor. View tensor shares the same underlying data
with its base tensor. Supporting ``View`` avoids explicit data copy, thus allows us to do fast and memory efficient
reshaping, slicing and element-wise operations.

For example, to get a view of an existing tensor ``t``, you can call ``t.view(...)``.

::

    >>> t = torch.rand(4, 4)
    >>> b = t.view(2, 8)
    >>> t.storage().data_ptr() == b.storage().data_ptr()  # `t` and `b` share the same underlying data.
    True
    # Modifying view tensor changes base tensor as well.
    >>> b[0][0] = 3.14
    >>> t[0][0]
    tensor(3.14)

Since views share underlying data with its base tensor, if you edit the data
in the view, it will be reflected in the base tensor as well.

Typically a PyTorch op returns a new tensor as output, e.g. :meth:`~torch.Tensor.add`.
But in case of view ops, outputs are views of input tensors to avoid unncessary data copy.
No data movement occurs when creating a view, view tensor just changes the way
it interprets the same data. Taking a view of contiguous tensor could potentially produce a non-contiguous tensor.
Users should be pay additional attention as contiguity might have implicit performance impact.
:meth:`~torch.Tensor.transpose` is a common example.

::

    >>> base = torch.tensor([[0, 1],[2, 3]])
    >>> base.is_contiguous()
    True
    >>> t = base.transpose(0, 1)  # `t` is a view of `base`. No data movement happened here.
    # View tensors might be non-contiguous.
    >>> t.is_contiguous()
    False
    # To get a contiguous tensor, call `.contiguous()` to enforce
    # copying data when `t` is not contiguous.
    >>> c = t.contiguous()

For reference, here’s a full list of view ops in PyTorch:

- Basic slicing and indexing op, e.g. ``tensor[0, 2:, 1:7:2]`` returns a view of base ``tensor``, see note below.
- :meth:`~torch.Tensor.as_strided`
- :meth:`~torch.Tensor.diagonal`
- :meth:`~torch.Tensor.expand`
- :meth:`~torch.Tensor.narrow`
- :meth:`~torch.Tensor.permute`
- :meth:`~torch.Tensor.select`
- :meth:`~torch.Tensor.squeeze`
- :meth:`~torch.Tensor.transpose`
- :meth:`~torch.Tensor.t`
- :attr:`~torch.Tensor.T`
- :meth:`~torch.Tensor.unfold`
- :meth:`~torch.Tensor.unsqueeze`
- :meth:`~torch.Tensor.view`
- :meth:`~torch.Tensor.view_as`
- :meth:`~torch.Tensor.unbind`
- :meth:`~torch.Tensor.split`
- :meth:`~torch.Tensor.chunk`
- :meth:`~torch.Tensor.indices` (sparse tensor only)
- :meth:`~torch.Tensor.values`  (sparse tensor only)

.. note::
   When accessing the contents of a tensor via indexing, PyTorch follows Numpy behaviors
   that basic indexing returns views, while advanced indexing returns a copy.
   Assignment via either basic or advanced indexing is in-place. See more examples in
   `Numpy indexing documentation <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.

It's also worth mentioning a few ops with special behaviors:

- :meth:`~torch.Tensor.reshape` and :meth:`~torch.Tensor.reshape_as` can return either a view or new tensor, user code shouldn't rely on whether it's view or not.
- :meth:`~torch.Tensor.contiguous` returns **itself** if input tensor is already contiguous, otherwise it returns a new contiguous tensor by copying data.

For a more detailed walk-through of PyTorch internal implementation,
please refer to `ezyang's blogpost about PyTorch Internals <http://blog.ezyang.com/2019/05/pytorch-internals/>`_.

