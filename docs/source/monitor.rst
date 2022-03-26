torch.monitor
=============

.. warning::

    This module is a prototype release, and its interfaces and functionality may
    change without warning in future PyTorch releases.

``torch.monitor`` provides an interface for logging events and counters from
PyTorch.

The stat interfaces are designed to be used for tracking high level metrics that
are periodically logged out to be used for monitoring system performance. Since
the stats aggregate with a specific window size you can log to them from
critical loops with minimal performance impact.

For more infrequent events or values such as loss, accuracy, usage tracking the
event interface can be directly used.

Event handlers can be registered to handle the events and pass them to an
external event sink.

API Reference
-------------

.. automodule:: torch.monitor

.. autoclass:: torch.monitor.Aggregation
    :members:

.. autoclass:: torch.monitor.Stat
    :members:
    :special-members: __init__

.. autoclass:: torch.monitor.data_value_t
    :members:

.. autoclass:: torch.monitor.Event
    :members:
    :special-members: __init__

.. autoclass:: torch.monitor.EventHandlerHandle
    :members:

.. autofunction:: torch.monitor.log_event

.. autofunction:: torch.monitor.register_event_handler

.. autofunction:: torch.monitor.unregister_event_handler

.. autoclass:: torch.monitor.TensorboardEventHandler
    :members:
    :special-members: __init__
