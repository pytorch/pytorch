torch.xpu
===================================
.. automodule:: torch.xpu
.. currentmodule:: torch.xpu

.. autosummary::
    :toctree: generated
    :nosignatures:

    StreamContext
    current_device
    current_stream
    device
    device_count
    device_of
    get_arch_list
    get_device_capability
    get_device_name
    get_device_properties
    get_gencode_flags
    init
    is_available
    is_initialized
    set_device
    set_stream
    stream
    synchronize

Random Number Generator
-------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_rng_state
    get_rng_state_all
    initial_seed
    manual_seed
    manual_seed_all
    seed
    seed_all
    set_rng_state
    set_rng_state_all

Streams and events
------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    Event
    Stream


Memory management
-----------------
.. autosummary::
    :toctree: generated
    :nosignatures:

     empty_cache
     max_memory_allocated
     max_memory_reserved
     memory_allocated
     memory_reserved
     memory_stats
     memory_stats_as_nested_dict
     reset_accumulated_memory_stats
     reset_peak_memory_stats


.. This module needs to be documented. Adding here in the meantime
.. for tracking purposes
.. py:module:: torch.xpu.memory
.. py:module:: torch.xpu.random
.. py:module:: torch.xpu.streams
