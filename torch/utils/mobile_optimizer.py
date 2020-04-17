"""
This module contains utility method for mobile model optimization and lint.
"""

import torch


def optimize_for_mobile(scripted_model):
    """
    Args:
        scripted_model: An instance of torch script module with type of ScriptModule

    Returns:
        scripted_model: A new optimized torch script module
    """
    if not isinstance(scripted_model, torch.jit.ScriptModule):
        raise TypeError(
            'Got {}, but ScriptModule is expected.'.format(type(scripted_model)))

    optimized_cpp_module = torch._C._jit_pass_optimize_for_mobile(scripted_model._c)
    return torch.jit._recursive.wrap_cpp_module(optimized_cpp_module)
