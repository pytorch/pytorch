import os.path as _osp
import torch

from .throughput_benchmark import ThroughputBenchmark
from .cpp_backtrace import get_cpp_backtrace
from .backend_registration import rename_privateuse1_backend, generate_methods_for_privateuse1_backend
from . import deterministic
from . import collect_env

def set_module(obj, mod):
    """
    Set the module attribute on a python object for a given object for nicer printing
    """
    if not isinstance(mod, str):
        raise TypeError("The mod argument should be a string")
    obj.__module__ = mod

if torch._running_with_deploy():
    # not valid inside torch_deploy interpreter, no paths exists for frozen modules
    cmake_prefix_path = None
else:
    cmake_prefix_path = _osp.join(_osp.dirname(_osp.dirname(__file__)), 'share', 'cmake')
