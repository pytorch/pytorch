_is_fastpath_enabled: bool = True

def get_fastpath_enabled() -> bool:
    return _is_fastpath_enabled

def set_fastpath_enabled(value: bool) -> None:
    global _is_fastpath_enabled
    _is_fastpath_enabled = value
