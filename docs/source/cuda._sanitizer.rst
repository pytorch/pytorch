.. currentmodule:: torch.cuda._sanitizer

CUDA Stream Sanitizer
=====================

.. note::
    This is a prototype feature, which means it is at an early stage
    for feedback and testing, and its components are subject to change.

Overview
--------

.. automodule:: torch.cuda._sanitizer


Usage
------

Here is an example of a simple synchronization error in PyTorch:

::

    import torch

    a = torch.rand(4, 2, device="cuda")

    with torch.cuda.stream(torch.cuda.Stream()):
        torch.mul(a, 5, out=a)

The ``a`` tensor is initialized on the default stream and, without any synchronization
methods, modified on a new stream. The two kernels will run concurrently on the same tensor,
which might cause the second kernel to read uninitialized data before the first one was able
to write it, or the first kernel might overwrite part of the result of the second.
When this script is run on the commandline with:
::

    TORCH_CUDA_SANITIZER=1 python example_error.py

the following output is printed by CSAN:

::

    ============================
    CSAN detected a possible data race on tensor with data pointer 139719969079296
    Access by stream 94646435460352 during kernel:
    aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    writing to argument(s) self, out, and to the output
    With stack trace:
      File "example_error.py", line 6, in <module>
        torch.mul(a, 5, out=a)
      ...
      File "pytorch/torch/cuda/_sanitizer.py", line 364, in _handle_kernel_launch
        stack_trace = traceback.StackSummary.extract(

    Previous access by stream 0 during kernel:
    aten::rand(int[] size, *, int? dtype=None, Device? device=None) -> Tensor
    writing to the output
    With stack trace:
      File "example_error.py", line 3, in <module>
        a = torch.rand(10000, device="cuda")
      ...
      File "pytorch/torch/cuda/_sanitizer.py", line 364, in _handle_kernel_launch
        stack_trace = traceback.StackSummary.extract(

    Tensor was allocated with stack trace:
      File "example_error.py", line 3, in <module>
        a = torch.rand(10000, device="cuda")
      ...
      File "pytorch/torch/cuda/_sanitizer.py", line 420, in _handle_memory_allocation
        traceback.StackSummary.extract(

This gives extensive insight into the origin of the error:

- A tensor was incorrectly accessed from streams with ids: 0 (default stream) and 94646435460352 (new stream)
- The tensor was allocated by invoking ``a = torch.rand(10000, device="cuda")``
- The faulty accesses were caused by operators
    - ``a = torch.rand(10000, device="cuda")`` on stream 0
    - ``torch.mul(a, 5, out=a)`` on stream 94646435460352
- The error message also displays the schemas of the invoked operators, along with a note
  showing which arguments of the operators correspond to the affected tensor.

  - In the example, it can be seen that tensor ``a`` corresponds to arguments ``self``, ``out``
    and the ``output`` value of the invoked operator ``torch.mul``.

.. seealso::
    The list of supported torch operators and their schemas can be viewed
    :doc:`here <torch>`.

The bug can be fixed by forcing the new stream to wait for the default stream:

::

    with torch.cuda.stream(torch.cuda.Stream()):
        torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
        torch.mul(a, 5, out=a)

When the script is run again, there are no errors reported.

API Reference
-------------

.. autofunction:: enable_cuda_sanitizer

.. autofunction:: zip_arguments

.. autofunction:: zip_by_key
