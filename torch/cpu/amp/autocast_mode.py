import torch
import functools
import warnings

class autocast(object):
    def __init__(self, enabled=True, dtype=torch.bfloat16):
        supported_dtype = [torch.bfloat16]
        if dtype not in supported_dtype :
            warnings.warn("In CPU autocast, but the target dtype is not supported. Disable the autocast.")
            warnings.warn("CPU Autocast only support dtype of torch.bfloat16 currently.")
            enabled = False
            dtype = torch.bfloat16
        self._enabled = enabled
        self._dtype = dtype

    def __enter__(self):
        self.prev = torch.is_autocast_cpu_enabled()
        self.prev_dtype = torch.get_autocast_cpu_dtype()
        torch.set_autocast_cpu_enabled(self._enabled)
        torch.set_autocast_cpu_dtype(self._dtype)
        torch.autocast_increment_nesting()

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if torch.autocast_decrement_nesting() == 0:
            torch.clear_autocast_cache()
        torch.set_autocast_cpu_enabled(self.prev)
        torch.set_autocast_cpu_dtype(self.prev_dtype)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast
