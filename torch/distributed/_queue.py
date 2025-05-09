import pickle
import threading
from dataclasses import dataclass

import torch
import torch.distributed as dist


class TensorQueue:
    """
    TensorQueue implements a fully distributed queue on top of a specified
    ProcessGroup using the default global TCPStore for metadata exchange.
    """

    def __init__(self, key: str, group=None) -> None:
        self.group = group
        self.store = dist.distributed_c10d._get_default_store().clone()

        self.rank = dist.get_rank(group)
        self.global_key = f"{key}/global"
        self.local_key = f"{key}/{self.rank}"

    def set_timeout(self, timeout: float) -> None:
        self.store.set_timeout(timeout)

    def push(self, tensor: torch.Tensor) -> None:
        self.store.queue_push(self.global_key, self.local_key)

        remote_rank = int(self.store.queue_pop(self.local_key))
        dist.send(tensor, dst=remote_rank, group=self.group)

    def pop(self, tensor: torch.Tensor) -> None:
        remote_key = self.store.queue_pop(self.global_key)
        remote_rank = int(remote_key.rpartition(b"/")[2])
        self.store.queue_push(remote_key, str(self.rank))

        dist.recv(tensor, src=remote_rank, group=self.group)


@dataclass
class _StoreRequest:
    key: str
    write: bool
    response_key: str
    rank: int


class TensorStore:
    """
    TensorStore implements a fully distributed store on top of a specified
    ProcessGroup using the default global TCPStore for metadata exchange.
    """

    def __init__(self, key: str, group=None) -> None:
        self.group = group
        self.store = dist.PrefixStore(
            key,
            dist.distributed_c10d._get_default_store().clone(),
        )

        self.rank = dist.get_rank(group)
        self.request_key = f"req/{self.rank}"
        self.response_key = f"resp/{self.rank}"

        self.tensors = {}

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        store = self.store.clone()
        while True:
            req = pickle.loads(store.queue_pop(self.request_key))
            store.queue_push(req.response_key, str(self.rank))
            tensor = self.tensors[req.key]
            if req.write:
                dist.recv(tensor, src=req.rank, group=self.group)
            else:
                dist.send(tensor, dst=req.rank, group=self.group)

    def set_timeout(self, timeout: float) -> None:
        self.store.set_timeout(timeout)

    def register(self, key: str, tensor: torch.Tensor) -> None:
        self.tensors[key] = tensor
        self.store.set(key, self.request_key)

    def write(self, key: str, tensor: torch.Tensor) -> None:
        request_key = self.store.get(key)
        req = _StoreRequest(
            write=True, response_key=self.response_key, key=key, rank=self.rank
        )
        self.store.queue_push(request_key, pickle.dumps(req))

        remote_rank = int(self.store.queue_pop(self.response_key))
        dist.send(tensor, dst=remote_rank, group=self.group)

    def read(self, key: str, tensor: torch.Tensor) -> None:
        request_key = self.store.get(key)
        req = _StoreRequest(
            write=False, response_key=self.response_key, key=key, rank=self.rank
        )
        self.store.queue_push(request_key, pickle.dumps(req))

        remote_rank = int(self.store.queue_pop(self.response_key))
        dist.recv(tensor, src=remote_rank, group=self.group)
