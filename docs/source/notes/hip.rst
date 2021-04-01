.. _hip-semantics:

HIP (ROCm) semantics
====================

ROCm\ |trade| is AMD’s open source software platform for GPU-accelerated high
performance computing and machine learning. HIP is ROCm's C++ dialect designed
to ease conversion of CUDA applications to portable C++ code. HIP is used when
converting existing CUDA applications like PyTorch to portable C++ and for new
projects that require portability between AMD and NVIDIA.

.. _hip_as_cuda:

HIP Interfaces Reuse the CUDA Interfaces
----------------------------------------

PyTorch for HIP intentionally reuses the existing :mod:`torch.cuda` interfaces.
This helps to accelerate the porting of existing PyTorch code and models because
very few code changes are necessary, if any.

The example from :ref:`cuda-semantics` will work exactly the same for HIP::

    cuda = torch.device('cuda')     # Default HIP device
    cuda0 = torch.device('cuda:0')  # 'rocm' or 'hip' are not valid, use 'cuda'
    cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)

    x = torch.tensor([1., 2.], device=cuda0)
    # x.device is device(type='cuda', index=0)
    y = torch.tensor([1., 2.]).cuda()
    # y.device is device(type='cuda', index=0)

    with torch.cuda.device(1):
        # allocates a tensor on GPU 1
        a = torch.tensor([1., 2.], device=cuda)

        # transfers a tensor from CPU to GPU 1
        b = torch.tensor([1., 2.]).cuda()
        # a.device and b.device are device(type='cuda', index=1)

        # You can also use ``Tensor.to`` to transfer a tensor:
        b2 = torch.tensor([1., 2.]).to(device=cuda)
        # b.device and b2.device are device(type='cuda', index=1)

        c = a + b
        # c.device is device(type='cuda', index=1)

        z = x + y
        # z.device is device(type='cuda', index=0)

        # even within a context, you can specify the device
        # (or give a GPU index to the .cuda call)
        d = torch.randn(2, device=cuda2)
        e = torch.randn(2).to(cuda2)
        f = torch.randn(2).cuda(cuda2)
        # d.device, e.device, and f.device are all device(type='cuda', index=2)

.. _checking_for_hip:

Checking for HIP
----------------

Whether you are using PyTorch for CUDA or HIP, the result of calling
:meth:`~torch.cuda.is_available` will be the same. If you are using a PyTorch
that has been built with GPU support, it will return `True`. If you must check
which version of PyTorch you are using, refer to this example below::

    if torch.cuda.is_available() and torch.version.hip:
        # do something specific for HIP
    elif torch.cuda.is_available() and torch.version.cuda:
        # do something specific for CUDA

