
from .throughput_benchmark import ThroughputBenchmark

import os.path as _osp

# Set the module for a given object for nicer printing
def set_module(obj, mod):
    if not isinstance(mod, str):
        raise TypeError("The mod argument should be a string")
    obj.__module__ = mod

#: Path to folder containing CMake definitions for Torch package
cmake_prefix_path = _osp.join(_osp.dirname(_osp.dirname(__file__)), 'share', 'cmake')
