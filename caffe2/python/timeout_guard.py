## @package timeout_guard
# Module caffe2.python.timeout_guard





import contextlib
import threading
import os
import time
import signal
import logging
from future.utils import viewitems


'''
Sometimes CUDA devices can get stuck, 'deadlock'. In this case it is often
better just the kill the process automatically. Use this guard to set a
maximum timespan for a python call, such as RunNet(). If it does not complete
in time, process is killed.

Example usage:
    with timeout_guard.CompleteInTimeOrDie(10.0):
        core.RunNet(...)
'''


class WatcherThread(threading.Thread):

    def __init__(self, timeout_secs):
        threading.Thread.__init__(self)
        self.timeout_secs = timeout_secs
        self.completed = False
        self.condition = threading.Condition()
        self.daemon = True
        self.caller_thread = threading.current_thread()

    def run(self):
        started = time.time()
        self.condition.acquire()
        while time.time() - started < self.timeout_secs and not self.completed:
            self.condition.wait(self.timeout_secs - (time.time() - started))
        self.condition.release()
        if not self.completed:
            log = logging.getLogger("timeout_guard")
            log.error("Call did not finish in time. Timeout:{}s PID: {}".format(
                self.timeout_secs,
                os.getpid(),
            ))

            # First try dying cleanly, but in 10 secs, exit properly
            def forcequit():
                time.sleep(10.0)
                log.info("Prepared output, dumping threads. ")
                print("Caller thread was: {}".format(self.caller_thread))
                print("-----After force------")
                log.info("-----After force------")
                import sys
                import traceback
                code = []
                for threadId, stack in viewitems(sys._current_frames()):
                    if threadId == self.caller_thread.ident:
                        code.append("\n# ThreadID: %s" % threadId)
                        for filename, lineno, name, line in traceback.extract_stack(stack):
                            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
                            if line:
                                code.append("  %s" % (line.strip()))

                # Log also with logger, as it is comment practice to suppress print().
                print("\n".join(code))
                log.info("\n".join(code))
                log.error("Process did not terminate cleanly in 10 s, forcing")
                os.abort()

            forcet = threading.Thread(target=forcequit, args=())
            forcet.daemon = True
            forcet.start()
            print("Caller thread was: {}".format(self.caller_thread))
            print("-----Before forcing------")
            import sys
            import traceback
            code = []
            for threadId, stack in viewitems(sys._current_frames()):
                code.append("\n# ThreadID: %s" % threadId)
                for filename, lineno, name, line in traceback.extract_stack(stack):
                    code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
                    if line:
                        code.append("  %s" % (line.strip()))

            # Log also with logger, as it is comment practice to suppress print().
            print("\n".join(code))
            log.info("\n".join(code))
            os.kill(os.getpid(), signal.SIGINT)


@contextlib.contextmanager
def CompleteInTimeOrDie(timeout_secs):
    watcher = WatcherThread(timeout_secs)
    watcher.start()
    yield
    watcher.completed = True
    watcher.condition.acquire()
    watcher.condition.notify()
    watcher.condition.release()


def EuthanizeIfNecessary(timeout_secs=120):
    '''
    Call this if you have problem with process getting stuck at shutdown.
    It will kill the process if it does not terminate in timeout_secs.
    '''
    watcher = WatcherThread(timeout_secs)
    watcher.start()
