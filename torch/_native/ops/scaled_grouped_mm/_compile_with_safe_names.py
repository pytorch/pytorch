import threading


_PATCH_LOCK = threading.Lock()


def _compile_with_safe_names(compile_fn):
    import cutlass.cute.core as cute_core
    from cutlass.base_dsl import dsl as base_dsl

    def _safe_mangle_name(self, function_name, args, args_spec):
        if threading.get_ident() != owner_thread:
            return orig_mangle(self, function_name, args, args_spec)
        del args, args_spec
        return function_name

    def _safe_pretty_str(arg):
        if threading.get_ident() != owner_thread:
            return orig_pretty_str(arg)
        try:
            return orig_pretty_str(arg)
        except Exception:
            return "<dynamic>"

    with _PATCH_LOCK:
        owner_thread = threading.get_ident()
        orig_mangle = base_dsl.BaseDSL.mangle_name
        orig_pretty_str = cute_core.pretty_str
        base_dsl.BaseDSL.mangle_name = _safe_mangle_name
        cute_core.pretty_str = _safe_pretty_str
        try:
            return compile_fn()
        finally:
            base_dsl.BaseDSL.mangle_name = orig_mangle
            cute_core.pretty_str = orig_pretty_str
