```{eval-rst}
.. currentmodule:: torch.profiler
```

# torch.profiler

## Overview
```{eval-rst}
.. automodule:: torch.profiler
```

## API Reference
```{eval-rst}
.. autoclass:: torch.profiler._KinetoProfile
  :members:

.. autoclass:: torch.profiler.profile
  :members:

.. autoclass:: torch.profiler.ProfilerAction
  :members:

.. autoclass:: torch.profiler.ProfilerActivity
  :members:

.. autofunction:: torch.profiler.schedule

.. autofunction:: torch.profiler.tensorboard_trace_handler

.. autofunction:: torch.profiler.register_export_chrome_trace_callback

.. autofunction:: torch.profiler.unregister_export_chrome_trace_callback
```

## Intel Instrumentation and Tracing Technology APIs

```{eval-rst}
.. autofunction:: torch.profiler.itt.is_available

.. autofunction:: torch.profiler.itt.mark

.. autofunction:: torch.profiler.itt.range_push

.. autofunction:: torch.profiler.itt.range_pop
```

<!-- This module needs to be documented. Adding here in the meantime
for tracking purposes -->
```{eval-rst}
.. py:module:: torch.profiler.itt
.. py:module:: torch.profiler.profiler
.. py:module:: torch.profiler.python_tracer
```
