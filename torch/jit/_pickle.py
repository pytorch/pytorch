# These functions are referenced from the pickle archives produced by
# ScriptModule.save()

def build_intlist(data):
    return data


def build_tensorlist(data):
    return data


def build_doublelist(data):
    return data


def build_boollist(data):
    return data


def build_tensor_from_id(data):
    if isinstance(data, int):
        # just the id, can't really do anything
        return data


def restore_type_tag(value, type_str):
    # The type_ptr is used by the jit unpickler to restore the full static type
    # to container types like list when they are re-loaded, but this doesn't
    # matter for Python, so just return the plain value
    return value
