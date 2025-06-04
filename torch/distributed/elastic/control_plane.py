import os
from collections.abc import Generator
from contextlib import contextmanager, ExitStack

from torch.distributed.elastic.multiprocessing.errors import record


__all__ = [
    "worker_main",
]

TORCH_WORKER_SERVER_SOCKET = "TORCH_WORKER_SERVER_SOCKET"


@contextmanager
def _worker_server(socket_path: str) -> Generator[None, None, None]:
    from torch._C._distributed_c10d import _WorkerServer

    server = _WorkerServer(socket_path)
    try:
        yield
    finally:
        server.shutdown()


@record
@contextmanager
def worker_main() -> Generator[None, None, None]:
    """
    This is a context manager that wraps your main entry function. This combines
    the existing ``errors.record`` logic as well as a new ``_WorkerServer`` that
    exposes handlers via a unix socket specified by
    ``Torch_WORKER_SERVER_SOCKET``.

    Example

    ::

     @worker_main()
     def main():
         pass


     if __name__ == "__main__":
         main()

    """
    with ExitStack() as stack:
        socket_path = os.environ.get(TORCH_WORKER_SERVER_SOCKET)
        if socket_path is not None:
            stack.enter_context(_worker_server(socket_path))

        yield
