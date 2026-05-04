.. _MPS-Backend:

MPS backend
===========

:mod:`mps` device enables high-performance
training on GPU for macOS devices with Metal programming framework.  It
introduces a new device to map Machine Learning computational graphs and
primitives on highly efficient Metal Performance Shaders Graph framework and
tuned kernels provided by Metal Performance Shaders framework respectively.

The new MPS backend extends the PyTorch ecosystem and provides existing scripts
capabilities to setup and run operations on GPU.

To get started, simply move your Tensor and Module to the ``mps`` device:

.. code:: python

    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current macOS version is not 14.0+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

    else:
        mps_device = torch.device("mps")

        # Create a Tensor directly on the mps device
        x = torch.ones(5, device=mps_device)
        # Or
        x = torch.ones(5, device="mps")

        # Any operation happens on the GPU
        y = x * 2

        # Move your model to mps just like any other device
        model = YourFavoriteNet()
        model.to(mps_device)

        # Now every call runs on the GPU
        pred = model(x)

.. _mps-memory-pinning:

Pinned memory
-------------

On Apple Silicon systems with unified memory, CPU tensors can be placed in
MPS-pinned memory with :meth:`~torch.Tensor.pin_memory` or by passing
``pin_memory=True`` to :class:`~torch.utils.data.DataLoader`. Pinned CPU tensors
are backed by shared Metal buffers, which lets non-blocking transfers to
``mps`` tensors avoid an extra host staging copy.
