torch.nested
============

.. automodule:: torch.nested

Introduction
++++++++++++

.. warning::

  The PyTorch API of nested tensors is in prototype stage and will change in the near future.

NestedTensor allows the user to pack a list of Tensors into a single, efficient datastructure.

The only constraint on the input Tensors is that their dimension must match.

This enables more efficient metadata representations and operator coverage.

Construction is straightforward and involves passing a list of Tensors to the constructor.

>>> a, b = torch.arange(3), torch.arange(5) + 3
>>> a
tensor([0, 1, 2])
>>> b
tensor([3, 4, 5, 6, 7])
>>> nt = torch.nested_tensor([a, b])
>>> nt
nested_tensor([
  tensor([0, 1, 2]),
    tensor([3, 4, 5, 6, 7])
    ])

Data type and device can be chosen via the usual keyword arguments

>>> nt = torch.nested_tensor([a, b], dtype=torch.float32, device="cuda")
>>> nt
nested_tensor([
  tensor([0., 1., 2.], device='cuda:0'),
  tensor([3., 4., 5., 6., 7.], device='cuda:0')
])

unbind allows you to retrieve a view of the constituents.

>>> nt = torch.nested_tensor([a, b], dtype=torch.float32, device="cuda")
>>> nt
nested_tensor([
  tensor([0., 1., 2.], device='cuda:0'),
  tensor([3., 4., 5., 6., 7.], device='cuda:0')
])
>>> nt.unbind()
[tensor([0., 1., 2.], device='cuda:0'), tensor([3., 4., 5., 6., 7.], device='cuda:0')]

Nested tensor methods
+++++++++++++++++++++++++

The following Tensor methods are related to nested tensors:

.. currentmodule:: torch
.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.to_padded_tensor
