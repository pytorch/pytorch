from threading import Lock, Thread
from time import sleep, time
from typing import Any, Optional, Union


class Timer:
    """
    This measures how long we have gone since last receiving a event and if it is grater than a set interval, calls a function.
    """

    def __init__(
        self,
        duration: Union[int, float],  # Duration in seconds
        call: Any,  # Function to call when we expire
    ) -> None:
        # We don't start the background thread until we actually get an event.
        self.background_thread: Optional[Thread] = None
        self.last_called: Optional[float] = None
        self.duration = duration
        self.sleep_time = 60
        self.call = call
        self.exit = False

        # Technically GIL should ensure only one call, but let's be explicit here
        self.lock = Lock()

    def record_call(self) -> None:
        with self.lock:
            if self.background_thread is None:
                self.background_thread = Thread(target=self.check)
                self.background_thread.start()
            self.last_called = time()

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
                if self.last_called + self.duration >= time():
                    continue
                self.last_called = None
                self.background_thread = None

            # Releasing lock in case self.call() takes a very long time or is reentrant
            self.call()
            return
