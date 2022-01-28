torch.monitor
=============

.. warning::

    This module is a prototype release, and its interfaces and functionality may
    change without warning in future PyTorch releases.

``torch.monitor`` provides an interface for logging events and counters from
PyTorch.


API Reference
-------------

.. automodule:: torch.monitor

.. autoclass:: torch.monitor.Aggregation
    :members:

.. autoclass:: torch.monitor.Stat
    :members:

.. autoclass:: torch.monitor.IntervalStat
    :members:

.. autoclass:: torch.monitor.FixedCountStat
    :members:

.. autoclass:: torch.monitor.Event
    :members:

.. autoclass:: torch.monitor.PythonEventHandler
    :members:

.. autofunction:: torch.monitor.log_event

.. autofunction:: torch.monitor.register_event_handler

.. autofunction:: torch.monitor.unregister_event_handler
