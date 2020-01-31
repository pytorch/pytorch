import torch
import functools


class autocast(object):
    r"""
    Instances of autocast are intended to be used as context managers or decorators.  API still under discussion...
    """
    # nesting_depth tracks how deeply nested the context manager invocations are.
    # When we __exit__ out of the last nesting, we see that nesting_depth drops to zero
    # and use that as a cue to clear the autocast cache.
    # TODO:
    # nesting_depth is a class variable so if different threads create instances of autocast
    # and manipulate the value, it won't be thread safe.
    # Currently, the cache is thread local, so maybe I should move nesting_depth to a thread local
    # variable on the C++ side.
    nesting_depth = 0

    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        self.prev = torch.is_autocast_enabled()
        torch.set_autocast_enabled(self._enabled)
        self.nesting_depth += 1

    def __exit__(self, *args):
        torch.set_autocast_enabled(self.prev)
        self.nesting_depth -= 1
        # Drop the cache when we exit back to a level that's not guarded by any instance of autocast.
        if self.nesting_depth == 0:
            torch.clear_autocast_cache()
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast
