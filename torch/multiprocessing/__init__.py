import sys as _sys
import multiprocessing as _mp


class _TorchContext(object):
    def __init__(self, ctx=None, sharing_strategy=None):
        if ctx is None:
            ctx = _mp
        if sharing_strategy is None:
            if _sys.platform == 'darwin':
                sharing_strategy = 'file_system'
            else:
                sharing_strategy = 'file_descriptor'
        if sharing_strategy not in self.get_all_sharing_strategies():
            raise ValueError("invalid sharing_strategy '{}'".format(sharing_strategy))
        self._ctx = ctx
        self._sharing_strategy = sharing_strategy

    def get_sharing_strategy(self):
        return self._sharing_strategy

    @staticmethod
    def get_all_sharing_strategies():
        if _sys.platform == 'darwin':
            return {'file_system'}
        else:
            return {'file_descriptor', 'file_system'}

    def get_context(self, method=None, sharing_strategy=None):
        ctx = self._ctx
        if method is not None:
            ctx = self._ctx.get_context(method)
        return _TorchContext(ctx, sharing_strategy)

    def Queue(self, context=None, reducers=None):
        from .queue import Queue, FdQueue
        if context is None:
            context = self
        if self._sharing_strategy == 'file_descriptor':
            return FdQueue(context, reducers)
        elif self._sharing_strategy == 'file_system':
            return Queue(context, reducers)

    def Pool(self, *args, **kwargs):
        from .pool import Pool
        if _sys.version_info[0] == 3:
            kwargs.setdefault('context', self)
        return Pool(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def __dir__(self):
        entries = dir(self._ctx)
        entries += ['get_sharing_strategy', 'get_all_sharing_strategies']
        if 'get_context' not in entries:
            entries += ['get_context']
        return entries


_default_context = _TorchContext()
_globals = globals()
for key in dir(_default_context):
    if key[0] != '_':
        _globals[key] = getattr(_default_context, key)

from ._storage import _init_storage_sharing
from ._tensor import _init_tensor_sharing
_init_storage_sharing()
_init_tensor_sharing()
