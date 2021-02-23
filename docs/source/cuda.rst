torch.cuda
===================================

.. currentmodule:: torch.cuda

.. automodule:: torch.cuda
   :members:

Random Number Generator
-------------------------
.. autofunction:: get_rng_state
.. autofunction:: get_rng_state_all
.. autofunction:: set_rng_state
.. autofunction:: set_rng_state_all
.. autofunction:: manual_seed
.. autofunction:: manual_seed_all
.. autofunction:: seed
.. autofunction:: seed_all
.. autofunction:: initial_seed


Communication collectives
-------------------------

.. autofunction:: torch.cuda.comm.broadcast

.. autofunction:: torch.cuda.comm.broadcast_coalesced

.. autofunction:: torch.cuda.comm.reduce_add

.. autofunction:: torch.cuda.comm.scatter

.. autofunction:: torch.cuda.comm.gather

Streams and events
------------------

.. autoclass:: Stream
   :members:

.. autoclass:: Event
   :members:

Memory management
-----------------
.. autofunction:: empty_cache
.. autofunction:: list_gpu_processes
.. autofunction:: memory_stats
.. autofunction:: memory_summary
.. autofunction:: memory_snapshot
.. autofunction:: memory_allocated
.. autofunction:: max_memory_allocated
.. autofunction:: reset_max_memory_allocated
.. autofunction:: memory_reserved
.. autofunction:: max_memory_reserved
.. autofunction:: set_per_process_memory_fraction
.. FIXME The following doesn't seem to exist. Is it supposed to?
   https://github.com/pytorch/pytorch/issues/27785
   .. autofunction:: reset_max_memory_reserved
.. autofunction:: memory_cached
.. autofunction:: max_memory_cached
.. autofunction:: reset_max_memory_cached

NVIDIA Tools Extension (NVTX)
-----------------------------

.. autofunction:: torch.cuda.nvtx.mark
.. autofunction:: torch.cuda.nvtx.range_push
.. autofunction:: torch.cuda.nvtx.range_pop
