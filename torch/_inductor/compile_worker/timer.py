from collections.abc import Callable
from threading import Lock, Thread
from time import monotonic, sleep


class Timer:
    """
    This measures how long we have gone since last receiving an event and if it is greater than a set interval, calls a function.
    """

    def __init__(
        self,
        duration: int | float,  # Duration in seconds
        call: Callable[[], None],  # Function to call when we expire
    ) -> None:
        # We don't start the background thread until we actually get an event.
        self.background_thread: Thread | None = None
        self.last_called: float | None = None
        self.duration = duration
        self.sleep_time = duration / 2
        self.call = call
        self.exit = False

        self.lock = Lock()

    def record_call(self) -> None:
        with self.lock:
            if self.background_thread is None:
                self.background_thread = Thread(
                    target=self.check, daemon=True, name="subproc_worker_timer"
                )
                self.background_thread.start()
            self.last_called = monotonic()

    def quit(self) -> None:
        with self.lock:
            self.exit = True

    def check(self) -> None:
        while True:
            # We have to be sensitive on checking here, to avoid too much impact on cpu
            sleep(self.sleep_time)
            with self.lock:
                if self.exit:
                    return
                assert self.last_called is not None
                if self.last_called + self.duration >= monotonic():
                    continue
                self.last_called = None
                self.background_thread = None

            # Releasing lock in case self.call() takes a very long time or is reentrant
            self.call()
            return
