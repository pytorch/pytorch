import torch
import functools


class autocast(object):
    r"""
    Instances of autocast are intended to be used as context managers or decorators.  API still under discussion...
    """
    # Tracker to help drop the cache if we are popping back up to a level that's not guarded
    # by any instance of autocast.  I worry about the thread safety of this tracker.
    # Might have to move it from python to c++ with a dedicated thread_local.
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
        # Drop the cache if we are popping back up to a level that's not guarded by any instance of autocast.
        if self.nesting_depth == 0:
            torch.clear_autocast_cache()
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast
