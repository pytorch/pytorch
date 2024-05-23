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
    set_stream
    stream
    synchronize
    device
    DeferredMtiaCallError

Streams and events
------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    Event
    Stream
