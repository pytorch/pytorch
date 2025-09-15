"""
Minimal distributed logging to prevent log spew across ranks.

Only warnings and logging statements on rank 0 are emitted.
"""

import logging
import warnings
import torch


def _is_non_rank_zero():
    """Check if we should suppress output (non-rank-0 in distributed mode)."""
    return (torch.distributed.is_available() and 
            torch.distributed.is_initialized() and 
            torch.distributed.get_rank() != 0)


def _make_rank_zero_only(original_func):
    """Create a wrapper that only executes on rank 0."""
    def wrapper(*args, **kwargs):
        if _is_non_rank_zero():
            return
        return original_func(*args, **kwargs)
    return wrapper


def patch_logging_for_distributed(patch_print=False):
    """
    Patch warnings and logging to only emit on rank 0 in distributed mode.
    
    Args:
        patch_print: If True, also patch print() to only emit on rank 0.
                    Default False since print is often used for debugging.
    """
    original_warn = warnings.warn
    
    def distributed_safe_warn(message, category=None, stacklevel=1, source=None):
        if _is_non_rank_zero():
            return
        return original_warn(message, category, stacklevel + 1, source)
    
    # Patch warnings.warn
    warnings.warn = distributed_safe_warn
    
    # Patch logging module functions
    for method_name in ['debug', 'info', 'warning', 'warn', 'error', 'critical']:
        if hasattr(logging, method_name):
            original_func = getattr(logging, method_name)
            setattr(logging, method_name, _make_rank_zero_only(original_func))
    
    # Patch Logger class methods to catch logger instances
    for method_name in ['debug', 'info', 'warning', 'warn', 'error', 'critical']:
        if hasattr(logging.Logger, method_name):
            original_method = getattr(logging.Logger, method_name)
            setattr(logging.Logger, method_name, _make_rank_zero_only(original_method))
    
    # Optionally patch print
    if patch_print:
        import builtins
        original_print = builtins.print
        builtins.print = _make_rank_zero_only(original_print)