import torch


global_flag = False
delete_global_value = True
delete_twice_value = True
delete_global_then_read_value = True
store_then_delete_value = True
delete_then_store_value = True
store_delete_store_value = True
delete_two_a_value = True
delete_two_b_value = True


def set_flag_true():
    global global_flag
    global_flag = True


def set_flag_false():
    global global_flag
    global_flag = False


def delete_global_value_fn():
    global delete_global_value
    del delete_global_value


def delete_missing_global_value_fn():
    global delete_missing_global_value
    del delete_missing_global_value


def delete_twice_fn():
    global delete_twice_value
    del delete_twice_value
    # Deleting an already-deleted global raises NameError.
    del delete_twice_value  # noqa: F821


def delete_missing_then_store_fn():
    global delete_missing_then_store_value
    del delete_missing_then_store_value
    # Unreachable in eager, which raises on the line above; the store is here to
    # prove Dynamo does not replay it either.
    delete_missing_then_store_value = 5  # noqa: F841


def store_then_delete_missing_fn():
    global store_then_delete_missing_value
    store_then_delete_missing_value = 5
    del store_then_delete_missing_value


def delete_global_then_read_fn():
    global delete_global_then_read_value
    del delete_global_then_read_value
    # Reading a deleted global raises NameError.
    return delete_global_then_read_value  # noqa: F821


def store_then_delete_preexisting_fn():
    global store_then_delete_value
    store_then_delete_value = 5
    del store_then_delete_value


def delete_then_store_preexisting_fn():
    global delete_then_store_value
    del delete_then_store_value
    # Re-added after the delete, so CPython appends it at the end of the dict.
    delete_then_store_value = 5  # noqa: F841


def store_delete_store_preexisting_fn():
    global store_delete_store_value
    store_delete_store_value = 5
    del store_delete_store_value
    # Re-added after the delete, so CPython appends it at the end of the dict.
    store_delete_store_value = 7  # noqa: F841


def delete_two_globals_fn():
    global delete_two_a_value, delete_two_b_value
    del delete_two_a_value
    del delete_two_b_value


def store_delete_delete_fn():
    global store_delete_delete_value
    store_delete_delete_value = 1
    del store_delete_delete_value
    # Deleting an already-deleted global raises NameError.
    del store_delete_delete_value  # noqa: F821


def store_then_delete_tensor_fn():
    global store_then_delete_tensor_value
    store_then_delete_tensor_value = torch.ones(4)
    del store_then_delete_tensor_value


def store_then_delete_multi_stream_tensor_fn(x, y, device):
    global store_then_delete_multi_stream_tensor_value
    s = torch.Stream(device=device)
    e = torch.Event(device=device)
    store_then_delete_multi_stream_tensor_value = x
    z0 = store_then_delete_multi_stream_tensor_value + 1
    with s:
        z = torch.add(store_then_delete_multi_stream_tensor_value, y)
        e.record()
    e.wait()
    del store_then_delete_multi_stream_tensor_value
    return z0, z
