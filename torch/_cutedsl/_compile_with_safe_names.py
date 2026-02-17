def _compile_with_safe_names(compile_fn):
    # CuTeDSL pretty-printing can throw on ScaledBasis during
    # compilation-time naming/printing. Patch temporarily.
    import cutlass.cute.core as cute_core
    from cutlass.base_dsl import dsl as base_dsl

    def _safe_mangle_name(self, function_name, args, args_spec):
        del args, args_spec
        return function_name

    def _safe_pretty_str(arg):
        try:
            return orig_pretty_str(arg)
        except Exception:
            return "<dynamic>"

    orig_mangle = base_dsl.BaseDSL.mangle_name
    orig_pretty_str = cute_core.pretty_str
    base_dsl.BaseDSL.mangle_name = _safe_mangle_name
    cute_core.pretty_str = _safe_pretty_str
    try:
        return compile_fn()
    finally:
        base_dsl.BaseDSL.mangle_name = orig_mangle
        cute_core.pretty_str = orig_pretty_str
