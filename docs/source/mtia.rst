torch.mtia
===================================

The MTIA backend is implemented out of the tree, only interfaces are be defined here.

.. automodule:: torch.mtia
.. currentmodule:: torch.mtia

.. autosummary::
    :toctree: generated
    :nosignatures:

    StreamContext
    current_device
    current_stream
    default_stream
    device_count
    init
    is_available
    is_initialized
    memory_stats
    max_memory_allocated
    get_device_capability
    empty_cache
    record_memory_history
    snapshot
    set_device
    set_stream
    stream
    synchronize
    device
    set_rng_state
    get_rng_state
    DeferredMtiaCallError

Streams and events
------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    Event
    Stream
