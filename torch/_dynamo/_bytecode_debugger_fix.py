from . import bytecode_debugger


class _CodeIdSet(set[int]):
    def add(self, code):
        super().add(id(code))

    def __contains__(self, code):
        return super().__contains__(id(code))

    def discard(self, code):
        super().discard(id(code))


_original_init = bytecode_debugger._DebugContext.__init__


def _init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self._tracked_codes = _CodeIdSet()


bytecode_debugger._DebugContext.__init__ = _init

__all__ = []
