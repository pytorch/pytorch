"""Observers for the in-process CUPTI activity monitor.

Each observer registers the activity kinds it needs with the shared monitor
(``torch.profiler._cupti.monitor.CuptiMonitor()``) and consumes the decoded columns
delivered to it, aggregating or assembling whatever it exposes to callers.
``CuptiMonitorObserver`` is the shared base.
"""
