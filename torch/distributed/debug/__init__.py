import logging
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

logger: logging.Logger = logging.getLogger(__name__)

_WORKER_SERVER: _WorkerServer | None = None
_DEBUG_SERVER_PROC: multiprocessing.Process | None = None


def start_debug_server(port: int = 25999, worker_port: int = 0) -> None:
    """
    Start the debug server stack on all workers. The frontend debug server is
    only started on rank0 while the per rank worker servers are started on all
    ranks.

    This server provides an HTTP frontend that allows for debugging slow and
    deadlocked distributed jobs across all ranks simultaneously. This collects
    data such as stack traces, FlightRecorder events, and performance profiles.

    This depends on dependencies which are not installed by default.

    Dependencies:
    - Jinja2
    - aiohttp

    WARNING: This is intended to only be used in trusted network environments.
    The debug server is not designed to be secure and should not be exposed to
    the public internet. See SECURITY.md for more details.

    WARNING: This is an experimental feature and may change at any time.

    Args:
        port (int): The port to start the frontend debug server on.
        worker_port (int): The port to start the worker server on. Defaults to 0, which
            will cause the worker server to bind to an ephemeral port.
    """
    global _WORKER_SERVER, _DEBUG_SERVER_PROC

    assert _WORKER_SERVER is None, "debug server already started"
    assert _DEBUG_SERVER_PROC is None, "debug server already started"

    logger.info("Starting debug server on port %d", port)

    store = tcpstore_client()

    _WORKER_SERVER = _WorkerServer("::", worker_port)

    RANK = get_rank()
    store.set(f"rank{RANK}", f"http://{socket.gethostname()}:{_WORKER_SERVER.port}")

    from torch.distributed.debug._frontend import main

    if RANK == 0:
        _DEBUG_SERVER_PROC = multiprocessing.Process(
            target=main, args=(port,), daemon=True
        )
        _DEBUG_SERVER_PROC.start()


def stop_debug_server() -> None:
    """
    Shutdown the debug server and stop the frontend debug server process.
    """
    global _WORKER_SERVER, _DEBUG_SERVER_PROC

    assert _DEBUG_SERVER_PROC is not None
    assert _WORKER_SERVER is not None

    logger.info("Stopping debug server")

    _DEBUG_SERVER_PROC.terminate()
    _WORKER_SERVER.shutdown()
    _DEBUG_SERVER_PROC.join()

    _WORKER_SERVER = None
    _DEBUG_SERVER_PROC = None
