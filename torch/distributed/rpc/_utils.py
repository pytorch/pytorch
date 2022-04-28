from contextlib import contextmanager
from typing import cast
from . import api
from . import TensorPipeAgent

@contextmanager
def _group_membership_management(store, name, is_join):
    token_key = "RpcGroupManagementToken"
    my_token = f"Token{name}-{int(is_join)}"
    while True:
        # Retrieve token from store to signal start of rank join/leave critical section
        returned = store.compare_set(token_key, "", my_token).decode()
        if returned == my_token:
            # Yield to the function this context manager wraps
            yield
            # Finished, now exit and release token
            # Update from store to signal end of rank join/leave critical section
            store.set(token_key, "")
            # Other will wait for this token to be set before they execute
            store.set(my_token, "Done")
            break
        else:
            # token_name = returned.split("-")[0]
            # Store will wait for the token to be released
            store.wait([returned])

def _update_group_membership(worker_info, my_devices, reverse_device_map, is_join):
    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    ret = agent._update_group_membership(worker_info, my_devices, reverse_device_map, is_join)
    return ret
