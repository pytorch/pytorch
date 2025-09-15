"""
Distributed-safe logging utilities.

This module provides monkey patching for warnings and logging to prevent
log spew in distributed training scenarios by only emitting messages on rank 0.
"""

import functools
import logging
import warnings
from typing import Any, Callable, Optional, Set
import torch


class DistributedLoggingState:
    """Global state for distributed logging patches."""
    
    def __init__(self):
        self.is_patched = False
        self.warned_messages: Set[str] = set()
        self.original_warn: Optional[Callable] = None
        self.patched_loggers: Set[logging.Logger] = set()
        self.original_logger_methods = {}
        self.original_print: Optional[Callable] = None
        self.printed_messages: Set[str] = set()
    
    def is_rank_zero_or_non_distributed(self) -> bool:
        """Check if we should emit logs (rank 0 or not in distributed mode)."""
        return (not torch.distributed.is_available() or 
                not torch.distributed.is_initialized() or 
                torch.distributed.get_rank() == 0)


_distributed_logging_state = DistributedLoggingState()


def _create_distributed_safe_warn():
    """Create a distributed-safe replacement for warnings.warn."""
    original_warn = warnings.warn
    
    def distributed_warn(message, category=None, stacklevel=1, source=None):
        # Convert to string for consistent deduplication
        msg_key = f"{message}|{category}"
        
        # Skip if we've already warned about this
        if msg_key in _distributed_logging_state.warned_messages:
            return
            
        # Only warn on rank 0 in distributed mode
        if not _distributed_logging_state.is_rank_zero_or_non_distributed():
            return
            
        _distributed_logging_state.warned_messages.add(msg_key)
        return original_warn(message, category, stacklevel + 1, source)
    
    return distributed_warn


def _create_distributed_safe_logger_method(original_method, method_name: str):
    """Create a distributed-safe replacement for logger methods."""
    
    def distributed_log_method(self, message, *args, **kwargs):
        # Only log on rank 0 in distributed mode
        if not _distributed_logging_state.is_rank_zero_or_non_distributed():
            return
            
        return original_method(self, message, *args, **kwargs)
    
    return distributed_log_method


def _patch_logger(logger: logging.Logger):
    """Patch a specific logger to be distributed-safe."""
    if logger in _distributed_logging_state.patched_loggers:
        return
        
    # Patch common logging methods
    for method_name in ['debug', 'info', 'warning', 'warn', 'error', 'critical']:
        if hasattr(logger, method_name):
            original_method = getattr(logger, method_name)
            # Store original for potential restoration
            key = (id(logger), method_name)
            _distributed_logging_state.original_logger_methods[key] = original_method
            
            # Replace with distributed-safe version
            distributed_method = _create_distributed_safe_logger_method(original_method, method_name)
            setattr(logger, method_name, distributed_method.__get__(logger, type(logger)))
    
    _distributed_logging_state.patched_loggers.add(logger)


def _patch_root_logger():
    """Patch the root logger to affect all loggers by default."""
    root_logger = logging.getLogger()
    _patch_logger(root_logger)
    
    # Also patch the logging module functions directly
    for method_name in ['debug', 'info', 'warning', 'warn', 'error', 'critical']:
        if hasattr(logging, method_name):
            original_func = getattr(logging, method_name)
            _distributed_logging_state.original_logger_methods[method_name] = original_func
            
            def make_distributed_logging_func(orig_func):
                def distributed_logging_func(msg, *args, **kwargs):
                    if not _distributed_logging_state.is_rank_zero_or_non_distributed():
                        return
                    return orig_func(msg, *args, **kwargs)
                return distributed_logging_func
            
            setattr(logging, method_name, make_distributed_logging_func(original_func))


def _create_distributed_safe_print():
    """Create a distributed-safe replacement for print."""
    import builtins
    original_print = builtins.print
    
    def distributed_print(*args, **kwargs):
        # For print, we don't deduplicate - just check rank
        if not _distributed_logging_state.is_rank_zero_or_non_distributed():
            return
            
        return original_print(*args, **kwargs)
    
    return distributed_print


def patch_logging_for_distributed(patch_print: bool = False):
    """
    Patch warnings and logging to be distributed-safe.
    
    After calling this function:
    - warnings.warn() will only emit on rank 0 and deduplicate messages
    - logger.info/debug/warning/etc will only emit on rank 0
    - logging.info/debug/warning/etc will only emit on rank 0
    - print() will only emit on rank 0 (if patch_print=True)
    
    Args:
        patch_print: Whether to also patch print statements. Default False since
                    this can be too aggressive for debugging.
    
    This function is idempotent - calling it multiple times is safe.
    """
    if _distributed_logging_state.is_patched:
        return
        
    # Patch warnings.warn
    _distributed_logging_state.original_warn = warnings.warn
    warnings.warn = _create_distributed_safe_warn()
    
    # Patch logging
    _patch_root_logger()
    
    # Optionally patch print
    if patch_print:
        import builtins
        _distributed_logging_state.original_print = builtins.print
        builtins.print = _create_distributed_safe_print()
    
    _distributed_logging_state.is_patched = True


def unpatch_logging():
    """
    Restore original logging behavior.
    
    This is primarily for testing and debugging purposes.
    """
    if not _distributed_logging_state.is_patched:
        return
        
    # Restore warnings.warn
    if _distributed_logging_state.original_warn:
        warnings.warn = _distributed_logging_state.original_warn
        _distributed_logging_state.original_warn = None
    
    # Restore print
    if _distributed_logging_state.original_print:
        import builtins
        builtins.print = _distributed_logging_state.original_print
        _distributed_logging_state.original_print = None
    
    # Restore patched loggers
    for logger in _distributed_logging_state.patched_loggers:
        for method_name in ['debug', 'info', 'warning', 'warn', 'error', 'critical']:
            key = (id(logger), method_name)
            if key in _distributed_logging_state.original_logger_methods:
                original_method = _distributed_logging_state.original_logger_methods[key]
                setattr(logger, method_name, original_method)
                
    # Restore logging module functions
    for method_name in ['debug', 'info', 'warning', 'warn', 'error', 'critical']:
        if method_name in _distributed_logging_state.original_logger_methods:
            original_func = _distributed_logging_state.original_logger_methods[method_name]
            setattr(logging, method_name, original_func)
    
    # Clear state
    _distributed_logging_state.patched_loggers.clear()
    _distributed_logging_state.original_logger_methods.clear()
    _distributed_logging_state.warned_messages.clear()
    _distributed_logging_state.printed_messages.clear()
    _distributed_logging_state.is_patched = False


def is_distributed_logging_enabled() -> bool:
    """Check if distributed logging patches are currently active."""
    return _distributed_logging_state.is_patched


# Additional utilities for special cases
def force_log_on_all_ranks(message: str, level: int = logging.INFO):
    """
    Force a log message to appear on all ranks.
    
    Use this sparingly for truly critical messages that need to appear everywhere.
    """
    rank = torch.distributed.get_rank() if (torch.distributed.is_available() and 
                                           torch.distributed.is_initialized()) else 0
    prefixed_message = f"[Rank {rank}] {message}"
    
    # Temporarily bypass our patches
    if _distributed_logging_state.original_warn:
        # Use original logging if available
        original_logging_info = _distributed_logging_state.original_logger_methods.get('info', logging.info)
        original_logging_info(prefixed_message)
    else:
        # Fallback to print if logging is heavily patched
        print(prefixed_message)


def distributed_print(*args, rank_prefix: bool = True, **kwargs):
    """
    Print only on rank 0, with optional rank prefix.
    
    This is a convenience function that respects distributed settings
    without requiring global print patching.
    """
    if not _distributed_logging_state.is_rank_zero_or_non_distributed():
        return
        
    if rank_prefix and torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        prefix = f"[Rank {rank}] "
        if args:
            args = (prefix + str(args[0]),) + args[1:]
        else:
            args = (prefix,)
    
    # Use original print if available, otherwise current print
    print_func = _distributed_logging_state.original_print or print
    return print_func(*args, **kwargs)


def patch_specific_logger(logger_name: str):
    """
    Patch a specific logger by name to be distributed-safe.
    
    Useful for third-party libraries that you want to make distributed-safe
    without patching all logging globally.
    """
    logger = logging.getLogger(logger_name)
    _patch_logger(logger)


def get_distributed_logging_stats():
    """
    Get statistics about distributed logging.
    
    Returns a dict with information about what's been patched and suppressed.
    """
    return {
        'is_patched': _distributed_logging_state.is_patched,
        'unique_warnings_seen': len(_distributed_logging_state.warned_messages),
        'patched_loggers_count': len(_distributed_logging_state.patched_loggers),
        'current_rank': torch.distributed.get_rank() if (torch.distributed.is_available() and 
                                                        torch.distributed.is_initialized()) else 0,
        'is_rank_zero': _distributed_logging_state.is_rank_zero_or_non_distributed(),
    }