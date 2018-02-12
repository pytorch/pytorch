torch.utils.bottleneck
===============

.. currentmodule:: torch.utils.bottleneck

`torch.utils.bottleneck` is a tool that can be used as an initial step for
debugging bottlenecks in your program. It summarizes runs of your script with 
the Python profiler and PyTorch's autograd profiler. 

Run it on the command line with 

::

    python -m torch.utils.bottleneck /path/to/source/script.py

or run ``python -m torch.utils.bottleneck -h`` for more usage instructions.

.. warning::
    For ease of use and intepretability of results, `bottleneck` runs multi-GPU
    code on only one GPU device. Internally, it sets the environment variables
    ``CUDA_LAUNCH_BLOCKING=1`` and ``CUDA_VISIBLE_DEVICES=K``, where ``K = 0``
    by default.

.. warning::
    Because your script will be profiled, please ensure that it exits in a 
    finite amount of time.

For more complicated uses of the profilers (like in a multi-GPU case),
please see https://docs.python.org/3/library/profile.html
or :func:`torch.autograd.profiler.profile()` for more information. 


