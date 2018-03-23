torch.utils.bottleneck
===============

.. currentmodule:: torch.utils.bottleneck

`torch.utils.bottleneck` is a tool that can be used as an initial step for
debugging bottlenecks in your program. It summarizes runs of your script with 
the Python profiler and PyTorch's autograd profiler. 

Run it on the command line with 

::

    python -m torch.utils.bottleneck -- /path/to/source/script.py [args]

where [args] are any number of arguments to `script.py`, or run
``python -m torch.utils.bottleneck -h`` for more usage instructions.

.. warning::
    Because your script will be profiled, please ensure that it exits in a 
    finite amount of time.

.. warning::
    Due to the asynchronous nature of CUDA kernels, when running against
    CUDA code, the cProfile output and CPU-mode autograd profilers may
    not show correct timings. In this case, the CUDA-mode autograd
    profiler is better at assigning blame to the relevant operator(s).

For more complicated uses of the profilers (like in a multi-GPU case),
please see https://docs.python.org/3/library/profile.html
or :func:`torch.autograd.profiler.profile()` for more information. 
