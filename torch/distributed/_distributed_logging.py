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


def patch_logging_for_distributed():
    """
    Patch warnings and logging to only emit on rank 0 in distributed mode.
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
            
            def make_distributed_logging_func(orig_func):
                def distributed_logging_func(msg, *args, **kwargs):
                    if _is_non_rank_zero():
                        return
                    return orig_func(msg, *args, **kwargs)
                return distributed_logging_func
            
            setattr(logging, method_name, make_distributed_logging_func(original_func))