def _ensure_iterable(obj):
    if not isinstance(obj, tuple):
        return (obj,)
    return obj
