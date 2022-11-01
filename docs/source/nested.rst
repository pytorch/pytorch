torch.nested
============

.. automodule:: torch.nested

Introduction
++++++++++++

.. warning::

  The PyTorch API of nested tensors is in prototype stage and will change in the near future.

NestedTensor allows the user to pack a list of Tensors into a single, efficient datastructure.

The only constraint on the input Tensors is that their dimension must match.

This enables more efficient metadata representations and access to purpose built kernels.

One application of NestedTensors is to express sequential data in various domains.
While the conventional approach is to pad variable length sequences, NestedTensor
enables users to bypass padding. The API for calling operations on a nested tensor is no different
from that of a regular ``torch.Tensor``, which should allow seamless integration with existing models,
with the main difference being :ref:`construction of the inputs <construction>`.

As this is a prototype feature, the :ref:`operations supported <supported operations>` are still
limited. However, we welcome issues, feature requests and contributions. More information on contributing can be found
`on this wiki <https://github.com/pytorch/pytorch/wiki/NestedTensor-Backend>`_.

.. _construction:

Construction
++++++++++++

Construction is straightforward and involves passing a list of Tensors to the ``torch.nested.nested_tensor``
constructor.

>>> a, b = torch.arange(3), torch.arange(5) + 3
>>> a
tensor([0, 1, 2])
>>> b
tensor([3, 4, 5, 6, 7])
>>> nt = torch.nested.nested_tensor([a, b])
>>> nt
nested_tensor([
  tensor([0, 1, 2]),
    tensor([3, 4, 5, 6, 7])
    ])

Data type, device and whether gradients are required can be chosen via the usual keyword arguments.

>>> nt = torch.nested.nested_tensor([a, b], dtype=torch.float32, device="cuda", requires_grad=True)
>>> nt
nested_tensor([
  tensor([0., 1., 2.], device='cuda:0', requires_grad=True),
  tensor([3., 4., 5., 6., 7.], device='cuda:0', requires_grad=True)
], device='cuda:0', requires_grad=True)

In the vein of ``torch.as_tensor``, ``torch.nested.as_nested_tensor`` can be used to preserve autograd
history from the tensors passed to the constructor. For more information, refer to the section on
:ref:`constructor functions`.

In order to form a valid NestedTensor all the passed Tensors need to match in dimension, but none of the other attributes need to.

>>> a = torch.randn(3, 50, 70) # image 1
>>> b = torch.randn(3, 128, 64) # image 2
>>> nt = torch.nested.nested_tensor([a, b], dtype=torch.float32)
>>> nt.dim()
4

If one of the dimensions doesn't match, the constructor throws an error.

>>> a = torch.randn(50, 128) # text 1
>>> b = torch.randn(3, 128, 64) # image 2
>>> nt = torch.nested.nested_tensor([a, b], dtype=torch.float32)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: All Tensors given to nested_tensor must have the same dimension. Found dimension 3 for Tensor at index 1 and dimension 2 for Tensor at index 0.

Note that the passed Tensors are being copied into a contiguous piece of memory. The resulting
NestedTensor allocates new memory to store them and does not keep a reference.

At this moment we only support one level of nesting, i.e. a simple, flat list of Tensors. In the future
we can add support for multiple levels of nesting, such as a list that consists entirely of lists of Tensors.
Note that for this extension it is important to maintain an even level of nesting across entries so that the resulting NestedTensor
has a well defined dimension. If you have a need for this feature, please feel encouraged to open a feature request so that
we can track it and plan accordingly.

size
+++++++++++++++++++++++++

Even though a NestedTensor does not support ``.size()`` (or ``.shape``), it supports ``.size(i)`` if dimension i is regular.

>>> a = torch.randn(50, 128) # text 1
>>> b = torch.randn(32, 128) # text 2
>>> nt = torch.nested.nested_tensor([a, b], dtype=torch.float32)
>>> nt.size(0)
2
>>> nt.size(1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Given dimension 1 is irregular and does not have a size.
>>> nt.size(2)
128

If all dimensions are regular, the NestedTensor is intended to be semantically indistinguishable from a regular ``torch.Tensor``.

>>> a = torch.randn(20, 128) # text 1
>>> nt = torch.nested.nested_tensor([a, a], dtype=torch.float32)
>>> nt.size(0)
2
>>> nt.size(1)
20
>>> nt.size(2)
128
>>> torch.stack(nt.unbind()).size()
torch.Size([2, 20, 128])
>>> torch.stack([a, a]).size()
torch.Size([2, 20, 128])
>>> torch.equal(torch.stack(nt.unbind()), torch.stack([a, a]))
True

In the future we might make it easier to detect this condition and convert seamlessly.

Please open a feature request if you have a need for this (or any other related feature for that matter).

unbind
+++++++++++++++++++++++++

``unbind`` allows you to retrieve a view of the constituents.

>>> import torch
>>> a = torch.randn(2, 3)
>>> b = torch.randn(3, 4)
>>> nt = torch.nested.nested_tensor([a, b], dtype=torch.float32)
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

Note that ``nt.unbind()[0]`` is not a copy, but rather a slice of the underlying memory, which represents the first entry or constituent of the NestedTensor.

.. _constructor functions:

Nested tensor constructor and conversion functions
++++++++++++++++++++++++++++++++++++++++++++++++++

The following functions are related to nested tensors:

.. currentmodule:: torch.nested

.. autofunction:: nested_tensor
.. autofunction:: as_nested_tensor
.. autofunction:: to_padded_tensor

.. _supported operations:

Supported operations
++++++++++++++++++++++++++

In this section, we summarize the operations that are currently supported on
NestedTensor and any constraints they have.

.. csv-table::
   :header: "PyTorch operation",  "Constraints"
   :widths: 30, 55
   :delim: ;

   :func:`torch.matmul`;  "Supports matrix multiplication between two (>= 3d) nested tensors where
   the last two dimensions are matrix dimensions and the leading (batch) dimensions have the same size
   (i.e. no broadcasting support for batch dimensions yet)."
   :func:`torch.bmm`; "Supports batch matrix multiplication of two 3-d nested tensors."
   :func:`torch.nn.Linear`;  "Supports 3-d nested input and a dense 2-d weight matrix."
   :func:`torch.nn.functional.softmax`; "Supports softmax along all dims except dim=0."
   :func:`torch.nn.Dropout`; "Behavior is the same as on regular tensors."
   :func:`torch.relu`; "Behavior is the same as on regular tensors."
   :func:`torch.gelu`; "Behavior is the same as on regular tensors."
   :func:`torch.neg`; "Behavior is the same as on regular tensors."
   :func:`torch.add`; "Supports elementwise addition of two nested tensors.
   Supports addition of a scalar to a nested tensor."
   :func:`torch.mul`; "Supports elementwise multiplication of two nested tensors.
   Supports multiplication of a nested tensor by a scalar."
   :func:`torch.select`; "Supports selecting along ``dim=0`` only (analogously ``nt[i]``)."
   :func:`torch.clone`; "Behavior is the same as on regular tensors."
   :func:`torch.detach`; "Behavior is the same as on regular tensors."
   :func:`torch.unbind`; "Supports unbinding along ``dim=0`` only."
   :func:`torch.reshape`; "Supports reshaping with size of ``dim=0`` preserved (i.e. number of tensors nested cannot be changed).
   Unlike regular tensors, a size of ``-1`` here means that the existing size is inherited.
   In particular, the only valid size for a ragged dimension is ``-1``.
   Size inference is not implemented yet and hence for new dimensions the size cannot be ``-1``."
   :func:`torch.Tensor.reshape_as`; "Similar constraint as for ``reshape``."
   :func:`torch.transpose`; "Supports transposing of all dims except ``dim=0``."
   :func:`torch.Tensor.view`; "Rules for the new shape are similar to that of ``reshape``."
