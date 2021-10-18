import threading
import time

class LocalQueue():
    ops = 0
    stored = 0
    uid = 0
    empty = 0

    def __init__(self, name='unnamed'):
        self.items = []
        self.name = name
        self.uid = LocalQueue.uid
        LocalQueue.uid += 1

    def put(self, item, block=True):
        LocalQueue.ops += 1
        LocalQueue.stored += 1
        self.items.append(item)

    def get(self, block=True, timeout=0):
        # TODO(VitalyFedyunin): Add support of block and timeout arguments
        LocalQueue.ops += 1
        if not len(self.items):
            LocalQueue.empty += 1
            raise Exception('LocalQueue is empty')
        LocalQueue.stored -= 1
        return self.items.pop()


class ThreadingQueue():
    def __init__(self, name='unnamed'):
        self.lock = threading.Lock()
        self.items = []
        self.name = name

    def put(self, item, block=True):
        with self.lock:
            self.items.append(item)

    def get(self, block=True, timeout=0):
        # TODO(VitalyFedyunin): Add support of block and timeout arguments
        while True:
            with self.lock:
                if len(self.items) > 0:
                    return self.items.pop()
            if not block:
                raise Exception("Not available")
            # TODO(VitalyFedyunin): Figure out what to do if nothing in the queue
            time.sleep(0.000001)
