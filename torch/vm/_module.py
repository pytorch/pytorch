"""The class of the ``torch.vm`` module itself: a module that can be called,
so that ``torch.vm(fn, passes=...)`` wraps a function. It lives in a private
module, as ``torch.ops``'s class does: an object whose class sits in a public
module is taken for public API of that module's parent."""

import types
from collections.abc import Callable
from typing import Any


class CallableModule(types.ModuleType):
    def __call__(self, fn: Callable | None = None, /, **kwargs: Any) -> Callable:
        return self.vm(fn, **kwargs)
