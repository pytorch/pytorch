import os.path as _osp
import sys

from .throughput_benchmark import ThroughputBenchmark
from ._crash_handler import enable_minidumps, disable_minidumps, enable_minidumps_on_exceptions
from .cpp_backtrace import get_cpp_backtrace
from .backend_registration import rename_privateuse1_backend

# Set the module for a given object for nicer printing
def set_module(obj, mod):
    if not isinstance(mod, str):
        raise TypeError("The mod argument should be a string")
    obj.__module__ = mod

if sys.executable == "torch_deploy":
    # not valid inside torch_deploy interpreter, no paths exists for frozen modules
    cmake_prefix_path = None
else:
    cmake_prefix_path = _osp.join(_osp.dirname(_osp.dirname(__file__)), 'share', 'cmake')
