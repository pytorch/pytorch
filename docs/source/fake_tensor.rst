Fake Tensor
===========

.. currentmodule:: torch.subclasses

Fake tensors are tensors that have all the metadata of a real tensor (shape, dtype, device, strides)
but no actual data. They are useful for:

- **Shape inference**: Determining output shapes without running actual computation
- **Compilation**: Running compiler passes that need tensor metadata without memory allocation
- **Debugging**: Testing tensor operation behavior without using GPU memory

For a detailed guide on using FakeTensor with torch.compile, see the
`FakeTensor and torch.compile user guide <https://pytorch.org/docs/stable/torch.compiler_fake_tensor.html>`_.

API Reference
-------------

FakeTensorMode
~~~~~~~~~~~~~~

.. autoclass:: FakeTensorMode
    :members: from_tensor
    :show-inheritance:
    :special-members: __enter__, __exit__

FakeTensor
~~~~~~~~~~

.. autoclass:: FakeTensor
    :members: fake_device, fake_mode
    :show-inheritance:

Exceptions
~~~~~~~~~~

.. autoexception:: UnsupportedFakeTensorException
    :show-inheritance:

.. autoexception:: DynamicOutputShapeException
    :show-inheritance:

Utilities
~~~~~~~~~

.. autofunction:: torch.subclasses.fake_tensor.unset_fake_temporarily

Related Topics
--------------

- `torch.compile documentation <https://pytorch.org/docs/stable/torch.compiler.html>`_
- `FakeTensor user guide <https://pytorch.org/docs/stable/torch.compiler_fake_tensor.html>`_
