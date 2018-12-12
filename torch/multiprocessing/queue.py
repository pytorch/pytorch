import sys
import io
import multiprocessing
from multiprocessing.reduction import ForkingPickler
import pickle

import multiprocessing.queues as mq


class ConnectionWrapper(object):
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler to
    serialize objects"""

    def __init__(self, conn):
        self.conn = conn

    def send(self, obj):
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        if 'conn' in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, 'conn'))


class Queue(mq.Queue):

    def __init__(self, *args, **kwargs):
        if 'ctx' not in kwargs:
            kwargs['ctx'] = multiprocessing.get_context()
        super(Queue, self).__init__(*args, **kwargs)
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv


class SimpleQueue(mq.SimpleQueue):

    def __init__(self, *args, **kwargs):
        if 'ctx' not in kwargs:
            kwargs['ctx'] = multiprocessing.get_context()
        super(SimpleQueue, self).__init__(*args, **kwargs)

    def _make_methods(self):
        if not isinstance(self._reader, ConnectionWrapper):
            self._reader = ConnectionWrapper(self._reader)
            self._writer = ConnectionWrapper(self._writer)
        super(SimpleQueue, self)._make_methods()
