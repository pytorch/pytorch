.. currentmodule:: torch.futures

.. _futures-docs:

torch.futures
=============

This package provides a :class:`~torch.futures.Future` type that encapsulates
an asynchronous execution and a set of utility functions to simplify operations
on :class:`~torch.futures.Future` objects. Currently, the
:class:`~torch.futures.Future` type is primarily used by the
:ref:`distributed-rpc-framework`.

.. automodule:: torch.futures

.. autoclass:: Future
    :inherited-members:

.. autofunction:: collect_all
.. autofunction:: wait_all
