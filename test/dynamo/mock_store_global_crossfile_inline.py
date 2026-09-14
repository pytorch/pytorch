import torch


global_flag = False
delete_value = True


def set_flag_true():
    global global_flag
    global_flag = True


def set_flag_false():
    global global_flag
    global_flag = False


def delete_value_fn():
    # Used for both the present-at-trace and absent-at-trace cases; the test
    # chooses by setting `delete_value` or removing it.
    global delete_value
    del delete_value


def store_then_delete_missing_fn():
    global store_then_delete_missing_value
    store_then_delete_missing_value = 5
    del store_then_delete_missing_value


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
