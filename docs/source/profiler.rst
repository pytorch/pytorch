.. currentmodule:: torch.profiler

torch.profiler
==============

Overview
--------
.. automodule:: torch.profiler


API Reference
-------------

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

Intel Instrumentation and Tracing Technology APIs
-------------------------------------------------

.. autofunction:: torch.profiler.itt.is_available

.. autofunction:: torch.profiler.itt.mark

.. autofunction:: torch.profiler.itt.range_push

.. autofunction:: torch.profiler.itt.range_pop

.. This module needs to be documented. Adding here in the meantime
.. for tracking purposes
.. py:module:: torch.profiler.itt
.. py:module:: torch.profiler.profiler
.. py:module:: torch.profiler.python_tracer