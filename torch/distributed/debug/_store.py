import os

import torch.distributed as dist


def get_rank() -> int:
    return int(os.environ["RANK"])


def get_world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def tcpstore_client() -> dist.Store:
    MASTER_ADDR = os.environ["MASTER_ADDR"]
    MASTER_PORT = int(os.environ["MASTER_PORT"])

    store = dist.TCPStore(
        host_name=MASTER_ADDR,
        port=MASTER_PORT,
        is_master=False,
    )
    store = dist.PrefixStore("debug_server", store)
    return store
