def build_intlist(data):
    return data


def build_tensor_from_id(data):
    if isinstance(data, int):
        # just the id, can't really do anything
        return data
