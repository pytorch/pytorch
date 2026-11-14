"""In-process CUPTI activity collection for torch.profiler.

``core`` owns the GIL-free CUPTI activity buffer collection (backed by the
native ``CuspyBuffers``); ``trace`` builds Chrome traces from it.
(The multiplexer + observers land here next.)
"""
