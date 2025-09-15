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
            
            def make_distributed_logging_func(orig_func):
                def distributed_logging_func(msg, *args, **kwargs):
                    if _is_non_rank_zero():
                        return
                    return orig_func(msg, *args, **kwargs)
                return distributed_logging_func
            
            setattr(logging, method_name, make_distributed_logging_func(original_func))
    
    # Patch Logger class methods to catch logger instances
    original_logger_methods = {}
    for method_name in ['debug', 'info', 'warning', 'warn', 'error', 'critical']:
        if hasattr(logging.Logger, method_name):
            original_method = getattr(logging.Logger, method_name)
            original_logger_methods[method_name] = original_method
            
            def make_distributed_logger_method(orig_method):
                def distributed_logger_method(self, msg, *args, **kwargs):
                    if _is_non_rank_zero():
                        return
                    return orig_method(self, msg, *args, **kwargs)
                return distributed_logger_method
            
            setattr(logging.Logger, method_name, make_distributed_logger_method(original_method))
    
    # Optionally patch print
    if patch_print:
        import builtins
        original_print = builtins.print
        
        def distributed_safe_print(*args, **kwargs):
            if _is_non_rank_zero():
                return
            return original_print(*args, **kwargs)
        
        builtins.print = distributed_safe_print