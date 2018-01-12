.. _cuda-semantics:

CUDA semantics
==============

:mod:`torch.cuda` is used to set up and run CUDA operations. It keeps track of
the currently selected GPU, and all CUDA tensors you allocate will by default be
created on that device. The selected device can be changed with a
:any:`torch.cuda.device` context manager.

However, once a tensor is allocated, you can do operations on it irrespective
of the selected device, and the results will be always placed in on the same
device as the tensor.

Cross-GPU operations are not allowed by default, with the only exception of
:meth:`~torch.Tensor.copy_`. Unless you enable peer-to-peer memory access, any
attempts to launch ops on tensors spread across different devices will raise an
error.

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

Memory management
-----------------

PyTorch use a caching memory allocator to speed up memory allocations. This
allows fast memory deallocation without device synchronizations. However, the
unused memory managed by the allocator will still show as if used in
`nvidia-smi`. You can use :meth:`~torch.cuda.memory_allocated` and
:meth:`~torch.cuda.max_memory_allocated` to monitor memory occupied by
tensors, and use :meth:`~torch.cuda.memory_cached` and
:meth:`~torch.cuda.max_memory_cached` to monitor memory managed by the caching
allocator. Calling :meth:`~torch.cuda.empty_cache` can release all unused cached
memory from PyTorch so that those can be used by other GPU applications.


Best practices
--------------

Device-agnostic code
^^^^^^^^^^^^^^^^^^^^

Due to the structure of PyTorch, you may need to explicitly write
device-agnostic (CPU or GPU) code; an example may be creating a new tensor as
the initial hidden state of a recurrent neural network.

The first step is to determine whether the GPU should be used or not. A common
pattern is to use Python's ``argparse`` module to read in user arguments, and
have a flag that can be used to disable CUDA, in combination with
:meth:`~torch.cuda.is_available`. In the following, ``args.cuda`` results in a
flag that can be used to cast tensors and modules to CUDA if desired::

    import argparse
    import torch

    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

If modules or tensors need to be sent to the GPU, ``args.cuda`` can be used as
follows::

    x = torch.Tensor(8, 42)
    net = Network()
    if args.cuda:
      x = x.cuda()
      net.cuda()

When creating tensors, an alternative to the if statement is to have a default
datatype defined, and cast all tensors using that. An example when using a
dataloader would be as follows::

    dtype = torch.cuda.FloatTensor
    for i, x in enumerate(train_loader):
        x = Variable(x.type(dtype))

When working with multiple GPUs on a system, you can use the
``CUDA_VISIBLE_DEVICES`` environment flag to manage which GPUs are available to
PyTorch. As mentioned above, to manually control which GPU a tensor is created
on, the best practice is to use a :any:`torch.cuda.device` context manager::

    print("Outside device is 0")  # On device 0 (default in most scenarios)
    with torch.cuda.device(1):
        print("Inside device is 1")  # On device 1
    print("Outside device is still 0")  # On device 0

If you have a tensor and would like to create a new tensor of the same type on
the same device, then you can use the :meth:`~torch.Tensor.new` method, which
acts the same as a normal tensor constructor. Whilst the previously mentioned
methods depend on the current GPU context, :meth:`~torch.Tensor.new` preserves
the device of the original tensor.

This is the recommended practice when creating modules in which new
tensors/variables need to be created internally during the forward pass::

    x_cpu = torch.FloatTensor(1)
    x_gpu = torch.cuda.FloatTensor(1)
    x_cpu_long = torch.LongTensor(1)

    y_cpu = x_cpu.new(8, 10, 10).fill_(0.3)
    y_gpu = x_gpu.new(x_gpu.size()).fill_(-5)
    y_cpu_long = x_cpu_long.new([[1, 2, 3]])

If you want to create a tensor of the same type and size of another tensor, and
fill it with either ones or zeros, :meth:`~torch.ones_like` or
:meth:`~torch.zeros_like` are provided as convenient helper functions (which
also preserve device)::

    x_cpu = torch.FloatTensor(1)
    x_gpu = torch.cuda.FloatTensor(1)

    y_cpu = torch.ones_like(x_cpu)
    y_gpu = torch.zeros_like(x_gpu)


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

Most use cases involving batched inputs and multiple GPUs should default to
using :class:`~torch.nn.DataParallel` to utilize more than one GPU. Even with
the GIL, a single Python process can saturate multiple GPUs.

As of version 0.1.9, large numbers of GPUs (8+) might not be fully utilized.
However, this is a known issue that is under active development. As always,
test your use case.

There are significant caveats to using CUDA models with
:mod:`~torch.multiprocessing`; unless care is taken to meet the data handling
requirements exactly, it is likely that your program will have incorrect or
undefined behavior.
