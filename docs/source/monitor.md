# torch.monitor

```{warning}
This module is a prototype release, and its interfaces and functionality may
change without warning in future PyTorch releases.
```

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

## API Reference
```{eval-rst}
.. automodule:: torch.monitor
```

```{eval-rst}
.. autoclass:: torch.monitor.Aggregation
    :members:
```

```{eval-rst}
.. autoclass:: torch.monitor.Stat
    :members:
    :special-members: __init__
```

```{eval-rst}
.. autoclass:: torch.monitor.data_value_t
    :members:
```

```{eval-rst}
.. autoclass:: torch.monitor.Event
    :members:
    :special-members: __init__
```

```{eval-rst}
.. autoclass:: torch.monitor.EventHandlerHandle
    :members:
```

```{eval-rst}
.. autofunction:: torch.monitor.log_event
```

```{eval-rst}
.. autofunction:: torch.monitor.register_event_handler
```

```{eval-rst}
.. autofunction:: torch.monitor.unregister_event_handler
```

```{eval-rst}
.. autoclass:: torch.monitor.TensorboardEventHandler
    :members:
    :special-members: __init__
```
