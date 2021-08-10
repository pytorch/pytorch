from .profiler import profile, tensorboard_trace_handler
from . import _listen
import signal
import os

logdir = './log/default' if os.environ.get('PYTORCH_PROFILER_SERVICE_LOGDIR') is None else os.environ.get('PYTORCH_PROFILER_SERVICE_LOGDIR')
port = 5000 if os.environ.get('PYTORCH_PROFILER_SERVICE_PORT') is None else int(os.environ.get('PYTORCH_PROFILER_SERVICE_PORT'))

prof = profile(
    on_trace_ready=tensorboard_trace_handler(logdir),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
)

def enter(signum, frame):
    print("start profiling")
    prof.__enter__()

def exit(signum, frame):
    print("exit profiling")
    prof.__exit__(None, None, None)

signal.signal(signal.SIGUSR1, enter)
signal.signal(signal.SIGUSR2, exit)

listener = _listen.Listener(os.getpid(), port)
listener.open()
