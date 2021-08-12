import abc
import os
import queue
import threading


class ClosureHandler(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def run(self, closure):
        """Run closure function

        Args:
          closure: callable function to run
        """
        pass

    def run_all(self, closures):
        for closure in closures:
            self.run(closure)


class AsyncClosureHandler(ClosureHandler):
    """Handler for Asynchronous Step Closures

    Args:
      max_queue_size: The maximum length of the closure queue after which
        the training loop will block until closures are evaluated.
        By default, a reasonable limit of a maximum of 100 on the queue.
        This value can be set using the `LTC_MAX_ASYNC_QUEUE` environment
        variable.
    """

    def __init__(self, max_queue_size=100):
        super().__init__()
        self._closure_queue = queue.Queue(
            os.environ.get("LTC_MAX_ASYNC_QUEUE", max_queue_size))
        self._closure_exception = queue.Queue()
        self._closure_lock = threading.Lock()
        self._closure_event_loop_finished = threading.Event()
        self._closure_event_loop = None

    def start_event_loop(self):
        """Start closure event loop if not started"""
        if self._closure_event_loop is None:

            def event_loop():
                # Run loop until closure event is set and closure queue is empty
                while True:
                    try:
                        closure = self._closure_queue.get(block=True, timeout=3)
                        closure()
                        self._closure_queue.task_done()
                    except queue.Empty:
                        with self._closure_lock:
                            if self._closure_queue.empty():
                                self._closure_event_loop_finished.set()
                                return
                    except Exception as e:
                        self._closure_exception.put(e)
                        return

            self._closure_event_loop = threading.Thread(target=event_loop)
            self._closure_event_loop.start()

    def run(self, closure):
        with self._closure_lock:
            self._closure_queue.put(closure, block=True)
            if (self._closure_event_loop is None or
                    not self._closure_event_loop.is_alive()):
                try:
                    e = self._closure_exception.get(block=False)
                    raise RuntimeError(
                        "Cannot run asynchronous closure due to previously raised exception"
                    ) from e
                except queue.Empty:
                    self._closure_event_loop = None
                    self.start_event_loop()

        return
