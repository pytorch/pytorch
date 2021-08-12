from __future__ import print_function

import collections
import threading


class QueueBase(object):

    def __init__(self, maxsize=1024):
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._ready_cv = threading.Condition(self._lock)
        self._space_available_cv = threading.Condition(self._lock)
        self._close_read = False
        self._close_write = False

    def max_size(self):
        return self._maxsize

    def close(self):
        with self._lock:
            self._close_read = True
            self._close_write = True
            self._ready_cv.notify_all()
            self._space_available_cv.notify_all()

    def close_write(self):
        with self._lock:
            self._close_write = True
            self._ready_cv.notify_all()


class KeydQueue(QueueBase):

    def __init__(self, maxsize=1024):
        super(KeydQueue, self).__init__(maxsize=maxsize)
        self._items = dict()
        self._waited_keys = set()

    def put(self, key, item):
        with self._lock:
            # Wait for space available, unless there is a waiter for the incoming
            # key.
            while (len(self._items) >= self._maxsize and
                   key not in self._waited_keys and not self._close_read):
                self._space_available_cv.wait()
            if not self._close_read:
                self._items[key] = item
                if key in self._waited_keys:
                    self._ready_cv.notify_all()

    def get(self, key):
        with self._lock:
            while key not in self._items and not self._close_write:
                self._waited_keys.add(key)
                self._space_available_cv.notify_all()
                self._ready_cv.wait()
                self._waited_keys.discard(key)
            item = self._items.pop(key, None)
            if item is not None:
                self._space_available_cv.notify()
            return item


class Queue(QueueBase):

    def __init__(self, maxsize=1024):
        super(Queue, self).__init__(maxsize=maxsize)
        self._items = collections.deque()

    def put(self, item):
        with self._lock:
            while (len(self._items) >= self._maxsize and not self._close_read):
                self._space_available_cv.wait()
            if not self._close_read:
                self._items.append(item)
                self._ready_cv.notify()

    def get(self):
        with self._lock:
            while not self._items and not self._close_write:
                self._ready_cv.wait()
            item = self._items.popleft() if self._items else None
            if item is not None:
                self._space_available_cv.notify()
            return item
