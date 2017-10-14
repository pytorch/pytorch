import ctypes
import tempfile
import contextlib
from . import cudart, check_error


class cudaOutputMode(object):
    cudaKeyValuePair = ctypes.c_int(0)
    cudaCSV = ctypes.c_int(1)

    @staticmethod
    def for_key(key):
        if key == 'key_value':
            return cudaOutputMode.cudaKeyValuePair
        elif key == 'csv':
            return cudaOutputMode.cudaCSV
        else:
            raise RuntimeError("supported CUDA profiler output modes are: key_value and csv")

DEFAULT_FLAGS = [
    "gpustarttimestamp",
    "gpuendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "streamid",
    "enableonstart 0",
    "conckerneltrace",
]


def init(output_file, flags=None, output_mode='key_value'):
    flags = DEFAULT_FLAGS if flags is None else flags
    output_mode = cudaOutputMode.for_key(output_mode)
    with tempfile.NamedTemporaryFile(delete=True) as f:
        f.write(b'\n'.join(map(lambda f: f.encode('ascii'), flags)))
        f.flush()
        check_error(cudart().cudaProfilerInitialize(
            ctypes.c_char_p(f.name.encode('ascii')), ctypes.c_char_p(output_file.encode('ascii')), output_mode))


def start():
    check_error(cudart().cudaProfilerStart())


def stop():
    check_error(cudart().cudaProfilerStop())


@contextlib.contextmanager
def profile():
    try:
        start()
        yield
    finally:
        stop()
