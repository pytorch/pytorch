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

Data type and device can be chosen via the usual keyword arguments.

>>> nt = torch.nested_tensor([a, b], dtype=torch.float32, device="cuda")
>>> nt
nested_tensor([
  tensor([0., 1., 2.], device='cuda:0'),
  tensor([3., 4., 5., 6., 7.], device='cuda:0')
])

Note that the passed Tensors are being copied into a contiguous piece of memory. The resulting
NestedTensor allocates new memory to store them and does not keep a reference.

In order to form a valid NestedTensor the passed Tensors also all need to match in dimension.

At this moment we only support one level of nesting, i.e. a simple, flat list of Tensors. In the future
we can add support for multiple levels of nesting, such as a list that consists entirely of lists of Tensors.
Note that for this extension it is important to maintain an even level of nesting across entries so that the resulting NestedTensor
has a well defined dimension. If you have a need for this feature, please feel encourage to open a feature request so that
we can track it and plan accordingly.

unbind
+++++++++++++++++++++++++

unbind allows you to retrieve a view of the constituents.

>>> nt = torch.nested_tensor([a, b], dtype=torch.float32, device="cuda")
>>> nt
nested_tensor([
  tensor([0., 1., 2.], device='cuda:0'),
  tensor([3., 4., 5., 6., 7.], device='cuda:0')
])
>>> nt.unbind()
[tensor([0., 1., 2.], device='cuda:0'), tensor([3., 4., 5., 6., 7.], device='cuda:0')]

>>> import torch
>>> a = torch.randn(2, 3)
>>> b = torch.randn(3, 4)
>>> nt = torch.nested_tensor([a, b], dtype=torch.float32)
>>> nt
nested_tensor([
  tensor([[ 1.2286, -1.2343, -1.4842],
          [-0.7827,  0.6745,  0.0658]]),
  tensor([[-1.1247, -0.4078, -1.0633,  0.8083],
          [-0.2871, -0.2980,  0.5559,  1.9885],
          [ 0.4074,  2.4855,  0.0733,  0.8285]])
])
>>> nt.unbind()
(tensor([[ 1.2286, -1.2343, -1.4842],
        [-0.7827,  0.6745,  0.0658]]), tensor([[-1.1247, -0.4078, -1.0633,  0.8083],
        [-0.2871, -0.2980,  0.5559,  1.9885],
        [ 0.4074,  2.4855,  0.0733,  0.8285]]))
>>> nt.unbind()[0] is not a
True
>>> nt.unbind()[0].mul_(3)
tensor([[ 3.6858, -3.7030, -4.4525],
        [-2.3481,  2.0236,  0.1975]])
>>> nt
nested_tensor([
  tensor([[ 3.6858, -3.7030, -4.4525],
          [-2.3481,  2.0236,  0.1975]]),
  tensor([[-1.1247, -0.4078, -1.0633,  0.8083],
          [-0.2871, -0.2980,  0.5559,  1.9885],
          [ 0.4074,  2.4855,  0.0733,  0.8285]])
])

Note that nt.unbind()[0] is not a, but rather a slice of the underlying memory, which represents the first entry or constituent of the NestedTensor.

Nested tensor methods
+++++++++++++++++++++++++

The following Tensor methods are related to nested tensors:

.. currentmodule:: torch
.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.to_padded_tensor
