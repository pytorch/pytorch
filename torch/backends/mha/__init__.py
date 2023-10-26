_is_fastpath_enabled = True

def get_fastpath_enabled():
    return _is_fastpath_enabled

def set_fastpath_enabled(value):
    global _is_fastpath_enabled
    _is_fastpath_enabled = value
