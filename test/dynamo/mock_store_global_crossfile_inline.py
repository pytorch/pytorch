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
    global delete_value
    del delete_value


def read_delete_value_fn():
    return delete_value


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


delete_then_store_value = 1


def delete_then_store_value_fn():
    global delete_then_store_value
    del delete_then_store_value
    delete_then_store_value = 7  # noqa: F841


delete_then_store_multi = 1


def delete_then_store_multi_fn():
    global delete_then_store_multi, delete_then_store_multi_new
    del delete_then_store_multi
    delete_then_store_multi_new = 2
    delete_then_store_multi = 3  # noqa: F841


store_read_delete_store_read_value = 1


def store_read_delete_store_read_value_fn(x):
    global store_read_delete_store_read_value
    store_read_delete_store_read_value = x + 1
    y = store_read_delete_store_read_value * 2
    del store_read_delete_store_read_value
    store_read_delete_store_read_value = y + 5
    z = store_read_delete_store_read_value * 3
    return y + z


flip_value = 1


def delete_flip_value_fn():
    global flip_value
    del flip_value


def delete_missing_value_fn():
    global delete_missing_value
    del delete_missing_value
