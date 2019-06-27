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
