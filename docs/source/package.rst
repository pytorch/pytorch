.. currentmodule:: torch.package

torch.package
=============

.. warning::

    This module is experimental and has not yet been publicly released.

``torch.package`` adds support for creating hermetic packages containing arbitrary
PyTorch code. These packages can be saved, shared, used to load and execute models
at a later date or on a different machine, and can even be deployed to production using
``torch::deploy``.

This document contains tutorials, how-to guides, explanations, and an API reference that
will help you learn more about ``torch.package`` and how to use it.

Tutorials
---------
Packaging your first model
^^^^^^^^^^^^^^^^^^^^^^^^^^
A tutorial that guides you through packaging and unpackaging a simple model is available
`on Colab <https://colab.research.google.com/drive/1dWATcDir22kgRQqBg2X_Lsh5UPfC7UTK?usp=sharing>`_.
After completing this exercise, you will be familiar with the basic API for creating and using
Torch packages.

API Reference
-------------
.. autoclass:: torch.package.PackagingError

.. autoclass:: torch.package.EmptyMatchError

.. autoclass:: torch.package.PackageExporter
  :members:

  .. automethod:: __init__

.. autoclass:: torch.package.PackageImporter
  :members:

  .. automethod:: __init__

.. autoclass:: torch.package.Directory
  :members:
