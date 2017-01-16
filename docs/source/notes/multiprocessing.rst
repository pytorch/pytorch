Multiprocessing best practices
==============================

:mod:`torch.multiprocessing` is a drop in replacement for Python's
:mod:`python:multiprocessing` module. It supports the exact same operations,
but extends it, so that all tensors sent through a
:class:`python:multiprocessing.Queue`, will have their data moved into shared
memory and will only send a handle to another process.

.. note::

    When a :class:`~torch.autograd.Variable` is sent to another process, both
    the :attr:`Variable.data` and :attr:`Variable.grad.data` are going to be
    shared.

This allows to implement various training methods, like Hogwild, A3C, or any
others that require asynchronous operation.

Sharing CUDA tensors
--------------------

Sharing CUDA tensors between processes is supported only in Python 3, using
a ``spawn`` or ``forkserver`` start methods. :mod:`python:multiprocessing` in
Python 2 can only create subprocesses using ``fork``, and it's not supported
by the CUDA runtime.

.. warning::

    CUDA API requires that the allocation exported to other processes remains
    valid as long as it's used by them. You should be careful and ensure that
    CUDA tensors you shared don't go out of scope as long as it's necessary.
    This shouldn't be a problem for sharing model parameters, but passing other
    kinds of data should be done with care. Note that this restriction doesn't
    apply to shared CPU memory.


Best practices
--------------

Reuse buffers passed through a Queue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remember that each time you put a :class:`~torch.Tensor` into a
:class:`python:multiprocessing.Queue`, it has to be moved into shared memory.
If it's already shared, it is a no-op, otherwise it will incur an additional
memory copy that can slow down the whole process. Even if you have a pool of
processes sending data to a single one, make it send the buffers back - this
is nearly free and will let you avoid a copy when sending next batch.






