torch.cuda
===================================
.. automodule:: torch.cuda
.. currentmodule:: torch.cuda

.. autosummary::
    :toctree: generated
    :nosignatures:

    StreamContext
    can_device_access_peer
    current_blas_handle
    current_device
    current_stream
    cudart
    default_stream
    device
    device_count
    device_memory_used
    device_of
    get_arch_list
    get_device_capability
    get_device_name
    get_device_properties
    get_gencode_flags
    get_stream_from_external
    get_sync_debug_mode
    init
    ipc_collect
    is_available
    is_initialized
    is_tf32_supported
    memory_usage
    set_device
    set_stream
    set_sync_debug_mode
    stream
    synchronize
    utilization
    temperature
    power_draw
    clock_rate
    OutOfMemoryError

Random Number Generator
-------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_rng_state
    get_rng_state_all
    set_rng_state
    set_rng_state_all
    manual_seed
    manual_seed_all
    seed
    seed_all
    initial_seed


Communication collectives
-------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    comm.broadcast
    comm.broadcast_coalesced
    comm.reduce_add
    comm.scatter
    comm.gather

Streams and events
------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    Stream
    ExternalStream
    Event

Graphs (beta)
-------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_current_stream_capturing
    graph_pool_handle
    CUDAGraph
    graph
    make_graphed_callables

.. _cuda-memory-management-api:

Memory management
-----------------
.. autosummary::
    :toctree: generated
    :nosignatures:

     empty_cache
     get_per_process_memory_fraction
     list_gpu_processes
     mem_get_info
     memory_stats
     memory_summary
     memory_snapshot
     memory_allocated
     max_memory_allocated
     reset_max_memory_allocated
     memory_reserved
     max_memory_reserved
     set_per_process_memory_fraction
     memory_cached
     max_memory_cached
     reset_max_memory_cached
     reset_peak_memory_stats
     caching_allocator_alloc
     caching_allocator_delete
     get_allocator_backend
     CUDAPluggableAllocator
     change_current_allocator
     MemPool
     MemPoolContext

.. currentmodule:: torch.cuda.memory

.. autosummary::
    :toctree: generated
    :nosignatures:

    caching_allocator_enable

.. currentmodule:: torch.cuda
.. autoclass:: torch.cuda.use_mem_pool

.. FIXME The following doesn't seem to exist. Is it supposed to?
   https://github.com/pytorch/pytorch/issues/27785
   .. autofunction:: reset_max_memory_reserved

NVIDIA Tools Extension (NVTX)
-----------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    nvtx.mark
    nvtx.range_push
    nvtx.range_pop
    nvtx.range

Jiterator (beta)
-----------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    jiterator._create_jit_fn
    jiterator._create_multi_output_jit_fn

TunableOp
---------

Some operations could be implemented using more than one library or more than
one technique. For example, a GEMM could be implemented for CUDA or ROCm using
either the cublas/cublasLt libraries or hipblas/hipblasLt libraries,
respectively. How does one know which implementation is the fastest and should
be chosen? That's what TunableOp provides. Certain operators have been
implemented using multiple strategies as Tunable Operators. At runtime, all
strategies are profiled and the fastest is selected for all subsequent
operations.

See the :doc:`documentation <cuda.tunable>` for information on how to use it.

.. toctree::
    :hidden:

    cuda.tunable


Stream Sanitizer (prototype)
----------------------------

CUDA Sanitizer is a prototype tool for detecting synchronization errors between streams in PyTorch.
See the :doc:`documentation <cuda._sanitizer>` for information on how to use it.

.. toctree::
    :hidden:

    cuda._sanitizer


GPUDirect Storage (prototype)
-----------------------------

The APIs in ``torch.cuda.gds`` provide thin wrappers around certain cuFile APIs that allow
direct memory access transfers between GPU memory and storage, avoiding a bounce buffer in the CPU. See the
`cufile api documentation <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-io-api>`_
for more details.

These APIs can be used in versions greater than or equal to CUDA 12.6. In order to use these APIs, one must
ensure that their system is appropriately configured to use GPUDirect Storage per the
`GPUDirect Storage documentation <https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/contents.html>`_.

See the docs for :class:`~torch.cuda.gds.GdsFile` for an example of how to use these.

.. currentmodule:: torch.cuda.gds
.. autosummary::
    :toctree: generated
    :nosignatures:

    gds_register_buffer
    gds_deregister_buffer
    GdsFile


.. This module needs to be documented. Adding here in the meantime
.. for tracking purposes
.. py:module:: torch.cuda.comm
.. py:module:: torch.cuda.error
.. py:module:: torch.cuda.gds
.. py:module:: torch.cuda.graphs
.. py:module:: torch.cuda.jiterator
.. py:module:: torch.cuda.memory
.. py:module:: torch.cuda.nccl
.. py:module:: torch.cuda.nvtx
.. py:module:: torch.cuda.profiler
.. py:module:: torch.cuda.random
.. py:module:: torch.cuda.sparse
.. py:module:: torch.cuda.streams
