"""In-process CUPTI activity collection for torch.profiler.

``monitor`` owns the GIL-free CUPTI activity buffer collection (backed by the
native ``CuptiMonitorBuffers``); ``monitor_trace`` builds Chrome traces from it.
(The multiplexer + observers land here next.)
"""
