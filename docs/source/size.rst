torch.Size
===================================

:class:`torch.Size` is the result type of a call to :func:`torch.Tensor.size`. It describes the size of all dimensions
of the original tensor. As a subclass of :class:`tuple`, it supports common sequence operations like indexing and
length.


Example::

    >>> x = torch.ones(10, 20, 30)
    >>> s = x.size()
    >>> s
    torch.Size([10, 20, 30])
    >>> s[1]
    20
    >>> len(s)
    3



.. autoclass:: torch.Size
   :members:
   :undoc-members:
   :inherited-members:
