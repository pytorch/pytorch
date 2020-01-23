.. currentmodule:: torch

.. _tensor-view-doc:

Tensor View
=================

In general there're three kinds of PyTorch ops:

- out-of-place ops: output is a new tensor with brand new storage.
- in-place ops: output is the input tensor with modified content, in other words ``id(output) == id(input)``.
- view ops: output is a new tensor which is a view of input tensor storage.

Different tensor views on the same storage allows us to do the following things:

- Fast and memory efficient reshaping. For example to view any tensor a single 1D buffer, you can do `t.view(-1)`.
- Fast and memory efficient slicing. For example to access the 2nd channel of any 3D Tensor representing an image you can do `t.select(0, 1)`.
- Fast element-wise operations of unrelated values. If you have 20 parameters that are scalars. You can create a single Tensor that hold all of them, and each parameter is just a view into this Tensor. When you need to perform an operation on all of these, you can replace the for-loop by simply doing the operation on the full Tensor and all the scalar parameters will be updated automatically.

But it can give surprising results if you are expecting a brand new Tensor. User should understand and use view ops as needed.

For reference, here’s a list of all view ops in PyTorch:

- :meth:`~torch.Tensor.transpose`
- :meth:`~torch.Tensor.as_strided`
- :meth:`~torch.Tensor.diagonal`
- :meth:`~torch.Tensor.expand`
- :meth:`~torch.Tensor.narrow`
- :meth:`~torch.Tensor.permute`
- :meth:`~torch.Tensor.select`
- :meth:`~torch.Tensor.squeeze`
- :meth:`~torch.Tensor.t`
- :meth:`~torch.Tensor.unfold`
- :meth:`~torch.Tensor.unsqueeze`
- :meth:`~torch.Tensor.indices`
- :meth:`~torch.Tensor.values`
- :meth:`~torch.Tensor.narrow`
- :meth:`~torch.Tensor.view`
- :meth:`~torch.Tensor.view_as`
- :meth:`~torch.Tensor.unbind`
- :meth:`~torch.Tensor.split`
- :meth:`~torch.Tensor.chunk`
- Basic indexing returns views, while advanced indexing returns copys. Assignment with both basic and advanced indexing is in-place. See more in :ref:`indexing-and-view`.

The following ops can return either a view of existing storage or new storage,
user code shouldn't rely on whether it's view or not.

- :meth:`~torch.Tensor.contiguous`
- :meth:`~torch.Tensor.reshape`
- :meth:`~torch.Tensor.reshape_as`

Indexing and views
------------------

.. _indexing-and-views:

PyTorch follows `Numpy behaviors <https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/BasicIndexing.html>`_ for basic indexing and advance indexing, which can be concluded as follows:

- Accessing the contents of a tensor via basic indexing does not create a copy of those contents. Rather, a view
 of the same underlying storage is produced.
Examples::

    >>> input = torch.rand(4, 4)
    >>> input[0] # view of input's storage
    >>> input[0:1] # view of input's storage
    >>> input[..., 0] # view of input's storage

.. note::
    Definition: Basic Indexing
    Given an N-dimensional tensor `x`, `x[index]` invokes basic indexing whenever `index` is a tuple containing
    any combination of the following types of object:
    - integers
    - slice
    - ellipsis

- Accessing the contents of a tensor via advanced indexing creates a copy of those contents.
Examples::

    >>> input = torch.tensor([0, -1, -2, -3, -4, -5])
    # advanced indexing with an integer tensor
    >>> index = torch.tensor([2, 4, 0, 4])
    >>> input[index]
    tensor([-2, -4, 0, -4]) # returns a copy
    # advanced indexing with boolean tensor
    >>> input = torch.randn(3, 3)
    tensor([[ 1.7713, -0.1840, -1.7450],
            [ 0.9422,  1.0072,  0.7350],
            [ 0.2717,  0.3600,  1.5939]])
    >>> bool_index = input > 0
    >>> input[bool_index]
    tensor([1.7713, 0.9422, 1.0072, 0.7350, 0.2717, 0.3600, 1.5939]) # returns a copy

.. note::
    Definition: Advanced Indexing
    Given an N-dimensional tensor `x`, `x[index]` invokes basic indexing whenever `index`:
    - an integer-type or boolean-type tensor
    - a tuple with at least one sequence-type object as an element (e.g. a list, tuple)

- Assignment via both basic indexing and advanced indexing is in-place.
Examples::

    # assignment via advanced indexing
    >>> input = torch.randn(3, 3)
    tensor([[ 1.7713, -0.1840, -1.7450],
            [ 0.9422,  1.0072,  0.7350],
            [ 0.2717,  0.3600,  1.5939]])
    >>> input[input < 0] = 0
    >>> input
    tensor([[1.7713, 0.0000, 0.0000],
            [0.9422, 1.0072, 0.7350],
            [0.2717, 0.3600, 1.5939]])
    # augmented assignment via integer tensor
    >>> ind0 = torch.tensor([0, -1])
    >>> ind0 = torch.tensor([0, 1])
    >>> input[ind0, ind1]
    tensor([1.7713, 0.3600])
    >>> input[ind0, ind1] *= 100
    tensor([[177.1337,   0.0000,   0.0000],
            [  0.9422,   1.0072,   0.7350],
            [  0.2717,  35.9969,   1.5939]])
    # An augmented assignment will only be performed once on redundant entries.
    >>> y = torch.tensor([4, 6, 8])
    # y[0] is accessed three times and y[2] one time
    >>> y[[0, 0, 0, 2]]
    tensor([4, 4, 4, 8])
    >>> y[[0, 0, 0, 2]] += 1
    tensor([5, 6, 9])

