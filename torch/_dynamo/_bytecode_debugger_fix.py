import threading

from . import bytecode_debugger, eval_frame


class _CodeIdSet(set[tuple[int, object]]):
    def add(self, code):
        super().add((id(code), code))

    def __contains__(self, code):
        return super().__contains__((id(code), code))

    def discard(self, code):
        super().discard((id(code), code))


_original_init = bytecode_debugger._DebugContext.__init__
_install_lock = threading.Lock()


def _init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self._tracked_codes = _CodeIdSet()


def _optimized_module_bool(self):
    return bool(self._orig_mod)


def install() -> None:
    if (
        getattr(bytecode_debugger._DebugContext, "_native_neo_debugger_fix_installed", False)
        and getattr(eval_frame.OptimizedModule, "_native_neo_bool_fix_installed", False)
    ):
        return
    with _install_lock:
        if (
            getattr(
                bytecode_debugger._DebugContext,
                "_native_neo_debugger_fix_installed",
                False,
            )
            and getattr(
                eval_frame.OptimizedModule,
                "_native_neo_bool_fix_installed",
                False,
            )
        ):
            return
        if not getattr(
            bytecode_debugger._DebugContext,
            "_native_neo_debugger_fix_installed",
            False,
        ):
            bytecode_debugger._DebugContext.__init__ = _init
            bytecode_debugger._DebugContext._native_neo_debugger_fix_installed = True
        if not getattr(
            eval_frame.OptimizedModule,
            "_native_neo_bool_fix_installed",
            False,
        ):
            eval_frame.OptimizedModule.__bool__ = _optimized_module_bool
            eval_frame.OptimizedModule._native_neo_bool_fix_installed = True


install()

__all__ = ["install"]
