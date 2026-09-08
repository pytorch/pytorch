from . import bytecode_debugger, eval_frame


class _CodeIdSet(set[tuple[int, object]]):
    def add(self, code):
        super().add((id(code), code))

    def __contains__(self, code):
        return super().__contains__((id(code), code))

    def discard(self, code):
        super().discard((id(code), code))


_original_init = bytecode_debugger._DebugContext.__init__


def _init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self._tracked_codes = _CodeIdSet()


def _optimized_module_bool(self):
    return bool(self._orig_mod)


bytecode_debugger._DebugContext.__init__ = _init
eval_frame.OptimizedModule.__bool__ = _optimized_module_bool

__all__ = []
