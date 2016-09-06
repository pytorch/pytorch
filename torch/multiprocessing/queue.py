import os
import socket
import multiprocessing
from itertools import chain
from io import BytesIO

import torch
from .common import CustomizablePicklingQueue, ExtendedInitPickler, \
    ExtendedInitUnpickler
from ._storage import reduce_storage
from ._tensor import reduce_tensor


class Queue(CustomizablePicklingQueue):

    def __init__(self, context=None, reducers=None):
        if context is None:
            context = multiprocessing
        if reducers is None:
            reducers = {}

        for t in torch._tensor_classes:
            reducers.setdefault(t, reduce_tensor)
        for t in torch._storage_classes:
            reducers.setdefault(t, reduce_storage)

        super(Queue, self).__init__(context, reducers)


class FdQueue(Queue):

    def __init__(self, *args, **kwargs):
        super(FdQueue, self).__init__(*args, **kwargs)
        self._fd_reader, self._fd_writer = socket.socketpair(socket.AF_UNIX)

    def __del__(self):
        self._fd_reader.close()
        self._fd_writer.close()

    def _send(self, obj):
        buffer = BytesIO()
        pickler = ExtendedInitPickler(buffer, self._reducers)
        pickler.dump(obj)
        # We need a list of unique file descriptors
        fd_list = list(set(obj._get_shared_fd() for obj in pickler.extended_init))
        pickler.dump(fd_list)
        socket_fd = self._fd_writer.fileno()
        # TODO: send fds in batches
        for fd in fd_list:
            torch._C._sendfd(socket_fd, fd)
        self._writer.send_bytes(buffer.getvalue())

    def _load(self):
        buf = BytesIO(self._reader.recv_bytes())
        pickler = ExtendedInitUnpickler(buf)
        result = pickler.load()
        fd_list = pickler.load()
        socket_fd = self._fd_reader.fileno()
        fd_map = {fd: torch._C._recvfd(socket_fd) for fd in fd_list}
        for obj in pickler.extended_init:
            obj._open_shared_fd(fd_map)
        for new_fd in fd_map.values():
            os.close(new_fd)
        return result

