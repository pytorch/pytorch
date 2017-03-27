.. _cuda-semantics:

CUDA semantics
==============

:mod:`torch.cuda` keeps track of currently selected GPU, and all CUDA tensors
you allocate will be created on it. The selected device can be changed with a
:any:`torch.cuda.device` context manager.

However, once a tensor is allocated, you can do operations on it irrespectively
of your selected device, and the results will be always placed in on the same
device as the tensor.

Cross-GPU operations are not allowed by default, with the only exception of
:meth:`~torch.Tensor.copy_`. Unless you enable peer-to-peer memory accesses
any attempts to launch ops on tensors spread across different devices will
raise an error.

Below you can find a small example showcasing this::

    x = torch.cuda.FloatTensor(1)
    # x.get_device() == 0
    y = torch.FloatTensor(1).cuda()
    # y.get_device() == 0

    with torch.cuda.device(1):
        # allocates a tensor on GPU 1
        a = torch.cuda.FloatTensor(1)

        # transfers a tensor from CPU to GPU 1
        b = torch.FloatTensor(1).cuda()
        # a.get_device() == b.get_device() == 1

        c = a + b
        # c.get_device() == 1

        z = x + y
        # z.get_device() == 0

        # even within a context, you can give a GPU id to the .cuda call
        d = torch.randn(2).cuda(2)
        # d.get_device() == 2

Best practices
--------------

Use pinned memory buffers
^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning:

    This is an advanced tip. You overuse of pinned memory can cause serious
    problems if you'll be running low on RAM, and you should be aware that
    pinning is often an expensive operation.

Host to GPU copies are much faster when they originate from pinned (page-locked)
memory. CPU tensors and storages expose a :meth:`~torch.Tensor.pin_memory`
method, that returns a copy of the object, with data put in a pinned region.

Also, once you pin a tensor or storage, you can use asynchronous GPU copies.
Just pass an additional ``async=True`` argument to a :meth:`~torch.Tensor.cuda`
call. This can be used to overlap data transfers with computation.

You can make the :class:`~torch.utils.data.DataLoader` return batches placed in
pinned memory by passing ``pin_memory=True`` to its constructor.

.. _cuda-nn-dataparallel-instead:

Use nn.DataParallel instead of multiprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most use cases involving batched input and multiple GPUs should default to using
:class:`~torch.nn.DataParallel` to utilize more than one GPU. Even with the GIL,
a single python process can saturate multiple GPUs.

As of version 0.1.9, large numbers of GPUs (8+) might not be fully utilized.
However, this is a known issue that is under active development. As always,
test your use case.

There are significant caveats to using CUDA models with
:mod:`~torch.multiprocessing`; unless care is taken to meet the data handling
requirements exactly, it is likely that your program will have incorrect or
undefined behavior.
