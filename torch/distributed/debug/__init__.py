import multiprocessing
import socket

# import for registration side effect
import torch.distributed.debug._handlers  # noqa: F401
from torch._C._distributed_c10d import _WorkerServer
from torch.distributed.debug._store import get_rank, tcpstore_client


__all__ = [
    "start_debug_server",
    "stop_debug_server",
]

_WORKER_SERVER: _WorkerServer | None = None
_DEBUG_SERVER_PROC: multiprocessing.Process | None = None


def start_debug_server(port: int = 25999, worker_port: int = 0) -> None:
    global _WORKER_SERVER, _DEBUG_SERVER_PROC

    assert _WORKER_SERVER is None, "debug server already started"
    assert _DEBUG_SERVER_PROC is None, "debug server already started"

    store = tcpstore_client()

    _WORKER_SERVER = _WorkerServer("::", worker_port)

    RANK = get_rank()
    store.set(f"rank{RANK}", f"http://{socket.gethostname()}:{_WORKER_SERVER.port}")

    from torch.distributed.debug._flask import main

    if RANK == 0:
        _DEBUG_SERVER_PROC = multiprocessing.Process(
            target=main, args=(port,), daemon=True
        )
        _DEBUG_SERVER_PROC.start()


def stop_debug_server() -> None:
    global _WORKER_SERVER, _DEBUG_SERVER_PROC

    assert _DEBUG_SERVER_PROC is not None
    assert _WORKER_SERVER is not None

    _DEBUG_SERVER_PROC.terminate()
    _WORKER_SERVER.shutdown()
    _DEBUG_SERVER_PROC.join()

    _WORKER_SERVER = None
    _DEBUG_SERVER_PROC = None
