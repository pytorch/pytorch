from __future__ import absolute_import, division, print_function, unicode_literals

from .throughput_benchmark import ThroughputBenchmark

# Set the module for a given object for nicer printing
def set_module(obj, mod):
    if not isinstance(mod, str):
        raise TypeError("The mod argument should be a string")
    obj.__module__ = mod
