from .profiler import profile, tensorboard_trace_handler
from . import _listen
import signal
import os
import multiprocessing

#logdir = './log/default' if os.environ.get('PYTORCH_PROFILER_SERVICE_LOGDIR') is None else os.environ.get('PYTORCH_PROFILER_SERVICE_LOGDIR')
host = '0.0.0.0' if os.environ.get('PYTORCH_PROFILER_SERVICE_HOST') is None else int(os.environ.get('PYTORCH_PROFILER_SERVICE_HOST'))
port = 3180 if os.environ.get('PYTORCH_PROFILER_SERVICE_PORT') is None else int(os.environ.get('PYTORCH_PROFILER_SERVICE_PORT'))

prof: profile = None

proc_manager = multiprocessing.Manager()
shared_config = proc_manager.dict({
    'log_dir': './log/default',
    'record_shapes': True,
    'profile_memory': True,
    'with_stack': True,
    'with_flops': False
})

def enter(signum, frame):
    print("start profiling")
    global prof
    prof = profile(
        on_trace_ready=tensorboard_trace_handler(shared_config['log_dir']),
        record_shapes=shared_config['record_shapes'],
        profile_memory=shared_config['profile_memory'],
        with_stack=shared_config['with_stack'],
        with_flops=shared_config['with_flops']
    )
    prof.start()

def stop(signum, frame):
    print("exit profiling")
    prof.stop()

signal.signal(signal.SIGUSR1, enter)
signal.signal(signal.SIGUSR2, stop)

listener = _listen.Listener(os.getpid(), host, port, shared_config)
listener.open()
