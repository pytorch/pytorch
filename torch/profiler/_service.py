from . import _listen
import os

host = 'localhost' if os.environ.get('PYTORCH_PROFILER_SERVICE_HOST') is None else str(os.environ.get('PYTORCH_PROFILER_SERVICE_HOST'))
port = 3180 if os.environ.get('PYTORCH_PROFILER_SERVICE_PORT') is None else int(os.environ.get('PYTORCH_PROFILER_SERVICE_PORT'))

listener = _listen.Listener(host, port)
listener.open()
