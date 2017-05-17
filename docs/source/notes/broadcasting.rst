.. _broadcasting-semantics:

Broadcasting semantics
==============

Many PyTorch operations support :any:`NumPy Broadcasting Semantics <numpy.doc.broadcasting>`.

In short, if a PyTorch operation supports broadcast, then its Tensor arguments can be
automatically expanded to be of equal sizes (without making copies of the data).

General semantics
-----------------
Two tensors are "broadcastable" if the following rules hold:

- Each tensor has at least one dimension.
- When iterating over the dimension sizes, starting at the trailing dimension,
  the dimension sizes must either be equal, one of them is 1, or one of them
  does not exist.

For Example::

    >>> x=torch.FloatTensor(5,7,3)
    >>> y=torch.FloatTensor(5,7,3)
    # same shapes are always broadcastable

    >>> x=torch.FloatTensor()
    >>> y=torch.FloatTensor(2,2)
    # x and y are not broadcastable, because x does not have at least 1 dimension
    
    >>> x=torch.FloatTensor(5,1,4,1)
    >>> y=torch.FloatTensor(3,1,1)
    # x and y are broadcastable
    
    # but:
    >>> x=torch.FloatTensor(5,2,4,1)
    >>> y=torch.FloatTensor(3,1,1)
    # x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3

If two tensors :attr:`x`, :attr:`y` are "broadcastable", the resulting tensor size
is calculated as follows:

- If the number of dimensions of :attr:`x` and :attr:`y` are not equal, prepend 1
  to the dimensions of the tensor with fewer dimensions to make them equal length.
- Then, for each dimension size, the resulting dimension size is the max of the sizes of
  :attr:`x` and :attr:`y` along that dimension.

For Example::

    # can line up trailing dimensions to make reading easier
    >>> x=torch.FloatTensor(5,1,4,1)
    >>> y=torch.FloatTensor(  3,1,1)
    >>> (x+y).size()
    torch.Size([5, 3, 4, 1])

    # but not necessary:
    >>> x=torch.FloatTensor(1)
    >>> y=torch.FloatTensor(3,1,7)
    >>> (x+y).size()
    torch.Size([3, 1, 7])

    >>> x=torch.FloatTensor(5,2,4,1)
    >>> y=torch.FloatTensor(3,1,1)
    >>> (x+y).size()
    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1

In-place semantics
-----------------
One complication is that in-place operations do not allow the in-place tensor to change shape
as a result of the broadcast.

For Example::

    >>> x=torch.FloatTensor(5,3,4,1)
    >>> y=torch.FloatTensor(3,1,1)
    >>> (x.add_(y)).size()
    torch.Size([5, 3, 4, 1])

    # but:
    >>> x=torch.FloatTensor(1,3,1)
    >>> y=torch.FloatTensor(3,1,7)
    >>> (x.add_(y)).size()
    RuntimeError: The expanded size of the tensor (1) must match the existing size (7) at non-singleton dimension 2.
