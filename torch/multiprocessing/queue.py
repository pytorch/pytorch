import sys
import io
import multiprocessing
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
import pickle


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


class Queue(multiprocessing.queues.Queue):

    def __init__(self, *args, **kwargs):
        if hasattr(multiprocessing, "get_context"):
            kwargs['ctx'] = multiprocessing.get_context()
        super(Queue, self).__init__(*args, **kwargs)

        if sys.version_info < (3, 3):
            self._reader = ConnectionWrapper(self._reader)
            self._writer = ConnectionWrapper(self._writer)
            self._send = self._writer.send
            self._recv = self._reader.recv

        self.sig_shutdown = multiprocessing.Value('i', 0)
        self.put_lock = multiprocessing.Lock()
        self.get_lock = multiprocessing.Lock()

    def __setstate__(self, state):
        default_state, extra_state = state
        (self.sig_shutdown, self.put_lock, self.get_lock) = extra_state
        super(Queue, self).__setstate__(default_state)

    def __getstate__(self):
        return (super(Queue, self).__getstate__(),
                (self.sig_shutdown, self.put_lock, self.get_lock))

    def shutdown(self):
        with self.sig_shutdown.get_lock():
            with self.put_lock and self.get_lock:
                self.sig_shutdown.value = 1

    def is_shutdown(self):
        with self.sig_shutdown.get_lock():
            return self.sig_shutdown.value

    def qsize(self):
        if self.is_shutdown():
            return -1
        super(Queue, self).qsize(self)

    def empty(self):
        if self.is_shutdown():
            return True
        return super(Queue, self).empty()

    def full(self):
        if self.is_shutdown():
            return True
        return super(Queue, self).full()

    def put(self, obj, block=True, timeout=None):
        if self.is_shutdown():
            return False
        with self.put_lock:
            super(Queue, self).put(obj, block, timeout)
            return True

    def get(self, block=True, timeout=None):
        if self.is_shutdown():
            return None
        with self.get_lock:
            return super(Queue, self).get(block, timeout)

    def get_nowait(self):
        return self.get(False)

    def put_nowait(self, obj):
        return self.put(obj, block=False)

    def close(self):
        if self.is_shutdown():
            return
        return super(Queue, self).close()

    def join_thread(self):
        if self.is_shutdown():
            return
        with self.put_lock and self.get_lock:
            return super(Queue, self).join_thread()

    def cancel_join_thread(self):
        if self.is_shutdown():
            return
        return super(Queue, self).cancel_join_thread()


class SimpleQueue(multiprocessing.queues.SimpleQueue):

    def _make_methods(self):
        if not isinstance(self._reader, ConnectionWrapper):
            self._reader = ConnectionWrapper(self._reader)
            self._writer = ConnectionWrapper(self._writer)
        super(SimpleQueue, self)._make_methods()
