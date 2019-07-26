.. _threading-cpu-inference:

Threading and CPU inference
===========================

PyTorch allows using multiple CPU threads during model inference. The following
figure shows different levels of parallelism one would find in a typical
application:

.. image:: threading_cpu_inference.png
   :width: 75%

One or more inference threads execute model's forward pass on the given inputs.
Each inference thread invokes a JIT interpreter that executes the ops
of a model inline, one by one. A model can utilize a ``fork`` TorchScript
primitive to launch some of the ops in parallel, as an asynchronous task, and
synchronize with them later, using a future returned from ``fork``.

PyTorch uses a single thread pool for the inter-op parallelism, this thread pool
is shared by all inference tasks within the application process.

Each operator can, in turn, use multiple threads to execute work. This can
be useful in many cases, including element-wise ops on large tensors,
convolutions, GEMMs, embedding lookups and others.

PyTorch provides a set of parallel primitives (in ``ATen/Parallel.h``), such as
``at::parallel_for``, that can be used to implement parallel ops.
In addition to the functions in ``ATen/Parallel.h``, parallel ops can also
utilize external libraries, such as MKL and MKL-DNN, to execute work.
``ATen/Parallel.h``, MKL, MKL-DNN and other libraries typically use parallelization
libraries (e.g. OpenMP or TBB) to implement multithreading. Intra-op thread
management is delegated to the parallelization library and is separate from the
inter-op thread pool.


Build options
-------------

PyTorch supports OpenMP and TBB implementations of intra-op parallelism specified
with the build settings:

+------------+-----------------------+-----------------------------+----------------------------------------+
| Library    | Build Option          | Values                      | Notes                                  |
+============+=======================+=============================+========================================+
| ATen       | ``ATEN_THREADING``    | ``OMP`` (default), ``TBB``  |                                        |
+------------+-----------------------+-----------------------------+----------------------------------------+
| MKL        | ``MKL_THREADING``     | (same)                      | To enable MKL use ``BLAS=MKL``         |
+------------+-----------------------+-----------------------------+----------------------------------------+
| MKL-DNN    | ``MKLDNN_THREADING``  | (same)                      | To enable MKL-DNN use ``USE_MKLDNN=1`` |
+------------+-----------------------+-----------------------------+----------------------------------------+

Any of the ``TBB`` values above require ``USE_TBB=1`` build setting (default: off).
A separate setting ``USE_OPENMP=1`` (default: on) is required for OpenMP parallelism.


Runtime API
-----------

The following API is used to control threading:

+------------------------+-----------------------------------------------------------+--------------------------------------------------+---------------------------------------------------------+
| Type of parallelism    | Settings                                                  | ATen/Parallel API                                | Notes                                                   |
+========================+===========================================================+==================================================+=========================================================+
| Inter-op parallelism   | ``at::set_num_interop_threads``,                          | ``at::launch`` (C++) - launches an inter-op task | ``set`` functions can only be called once and only      |
|                        | ``at::get_num_interop_threads`` (C++)                     |                                                  | during the startup, before the actual operators running;|
|                        |                                                           |                                                  |                                                         |
|                        | ``set_num_interop_threads``,                              |                                                  |                                                         |
|                        | ``get_num_interop_threads`` (Python, :mod:`torch` module) |                                                  | Default number of threads: number of CPU cores;         |
+------------------------+-----------------------------------------------------------+--------------------------------------------------+---------------------------------------------------------+
| Intra-op parallelism   | ``at::set_num_threads``,                                  | ``at::parallel_for``                             | ``set`` functions can only be called once and only      |
|                        | ``at::get_num_threads`` (C++)                             | ``at::parallel_reduce`` (C++)                    | during the startup, before the actual operators running;|
|                        | ``set_num_threads``,                                      |                                                  |                                                         |
|                        | ``get_num_threads`` (Python, :mod:`torch` module)         | launching intra-op async tasks:                  | Default number of threads: number of CPU cores;         |
|                        |                                                           | ``at::intraop_launch``                           |                                                         |
|                        | Env. variables:                                           | ``at::intraop_launch_future`` (C++)              | Number of threads setting preference:                   |
|                        | ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS``               |                                                  | ``at::API`` > ``MKL_NUM_THREADS`` > ``OMP_NUM_THREADS`` |
|                        |                                                           |                                                  |                                                         |
+------------------------+-----------------------------------------------------------+--------------------------------------------------+---------------------------------------------------------+

.. note::

    OpenMP does not guarantee that a single per-process intra-op thread pool would be used.
    In fact, two different inter-op threads will likely use different OpenMP thread pools for intra-op work.
    Use TBB backend to guarantee that there's a single per-process intra-op thread pool of a given size.

.. note::
    ``parallel_info`` utility prints information about thread settings and can be used for debugging.
    Similar output can be also obtained in Python with ``torch.__config__.parallel_info()`` call.
