:orphan:

.. _multiprocessing-doc:

Multiprocessing package - torch.multiprocessing
===============================================

.. automodule:: torch.multiprocessing
.. currentmodule:: torch.multiprocessing

.. warning::

    If the main process exits abruptly (e.g. because of an incoming signal),
    Python's ``multiprocessing`` sometimes fails to clean up its children.
    It's a known caveat, so if you're seeing any resource leaks after
    interrupting the interpreter, it probably means that this has just happened
    to you.

Strategy management
-------------------

.. autofunction:: get_all_sharing_strategies
.. autofunction:: get_sharing_strategy
.. autofunction:: set_sharing_strategy


.. _multiprocessing-cuda-sharing-details:

Sharing CUDA tensors
--------------------

Sharing CUDA tensors between processes is supported only in Python 3, using
a ``spawn`` or ``forkserver`` start methods. :mod:`python:multiprocessing` in
Python 2 can only create subprocesses using ``fork``, and it's not supported
by the CUDA runtime.

Unlike CPU tensors, the sending process is required to keep the original tensor
as long as the receiving process retains a copy of the tensor. The refcounting is
implemented under the hood but requires users to follow the next best practices.

.. warning::
    If the consumer process dies abnormally to a fatal signal, the shared tensor
    could be forever kept in memory as long as the sending process is running.


1. Release memory ASAP in the consumer.

::

    ## Good
    x = queue.get()
    # do somethings with x
    del x

::

    ## Bad
    x = queue.get()
    # do somethings with x
    # do everything else (producer have to keep x in memory)

2. Keep producer process running until all consumers exits. This will prevent
the situation when the producer process releasing memory which is still in use
by the consumer.

::

    ## producer
    # send tensors, do something
    event.wait()


::

    ## consumer
    # receive tensors and use them
    event.set()

3. Don't pass received tensors.

::

    # not going to work
    x = queue.get()
    queue_2.put(x)


::

    # you need to create a process-local copy
    x = queue.get()
    x_clone = x.clone()
    queue_2.put(x_clone)


::

    # putting and getting from the same queue in the same process will likely end up with segfault
    queue.put(tensor)
    x = queue.get()


Sharing strategies
------------------

This section provides a brief overview into how different sharing strategies
work. Note that it applies only to CPU tensor - CUDA tensors will always use
the CUDA API, as that's the only way they can be shared.

File descriptor - ``file_descriptor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. note::

    This is the default strategy (except for macOS and OS X where it's not
    supported).

This strategy will use file descriptors as shared memory handles. Whenever a
storage is moved to shared memory, a file descriptor obtained from ``shm_open``
is cached with the object, and when it's going to be sent to other processes,
the file descriptor will be transferred (e.g. via UNIX sockets) to it. The
receiver will also cache the file descriptor and ``mmap`` it, to obtain a shared
view onto the storage data.

Note that if there will be a lot of tensors shared, this strategy will keep a
large number of file descriptors open most of the time. If your system has low
limits for the number of open file descriptors, and you can't raise them, you
should use the ``file_system`` strategy.

File system - ``file_system``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This strategy will use file names given to ``shm_open`` to identify the shared
memory regions. This has a benefit of not requiring the implementation to cache
the file descriptors obtained from it, but at the same time is prone to shared
memory leaks. The file can't be deleted right after its creation, because other
processes need to access it to open their views. If the processes fatally
crash, or are killed, and don't call the storage destructors, the files will
remain in the system. This is very serious, because they keep using up the
memory until the system is restarted, or they're freed manually.

To counter the problem of shared memory file leaks, :mod:`torch.multiprocessing`
will spawn a daemon named ``torch_shm_manager`` that will isolate itself from
the current process group, and will keep track of all shared memory allocations.
Once all processes connected to it exit, it will wait a moment to ensure there
will be no new connections, and will iterate over all shared memory files
allocated by the group. If it finds that any of them still exist, they will be
deallocated. We've tested this method and it proved to be robust to various
failures. Still, if your system has high enough limits, and ``file_descriptor``
is a supported strategy, we do not recommend switching to this one.

Spawning subprocesses
---------------------

.. note::

   Available for Python >= 3.4.

   This depends on the ``spawn`` start method in Python's
   ``multiprocessing`` package.

Spawning a number of subprocesses to perform some function can be done
by creating ``Process`` instances and calling ``join`` to wait for
their completion. This approach works fine when dealing with a single
subprocess but presents potential issues when dealing with multiple
processes.

Namely, joining processes sequentially implies they will terminate
sequentially. If they don't, and the first process does not terminate,
the process termination will go unnoticed. Also, there are no native
facilities for error propagation.

The ``spawn`` function below addresses these concerns and takes care
of error propagation, out of order termination, and will actively
terminate processes upon detecting an error in one of them.

.. autofunction:: spawn

.. class:: SpawnContext

   Returned by :func:`~spawn` when called with ``join=False``.

   .. automethod:: join
