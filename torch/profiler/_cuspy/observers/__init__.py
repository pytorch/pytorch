"""Observers for Cuspy, the in-process CUPTI activity collector.

Each observer registers the activity kinds it needs with the shared Cuspy singleton
(``torch.profiler._cuspy.core.Cuspy()``) and consumes the decoded columns
delivered to it, aggregating or assembling whatever it exposes to callers.
``CuspyObserver`` is the shared base.
"""
