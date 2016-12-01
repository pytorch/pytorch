import ctypes
import torch
import os
from . import cudart


SUCCESS = 0
ERROR_NOT_READY = 34


class CudaError(RuntimeError):
    def __init__(self, code):
        msg = cudart().cudaGetErrorString(code).decode('utf-8')
        super(CudaError, self).__init__('{0} ({1})'.format(msg, code))


def check_error(res):
    if res != SUCCESS:
        raise CudaError(res)


class Stream(torch._C._CudaStreamBase):
    def __new__(cls, device=-1, **kwargs):
        with torch.cuda.device(device):
            return super(Stream, cls).__new__(cls, **kwargs)

    def wait_event(self, event):
        check_error(cudart().cudaStreamWaitEvent(self, event, ctypes.c_int(0)))

    def wait_stream(self, stream):
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        if event is None:
            event = Event()
        check_error(cudart().cudaEventRecord(event, self))
        return event

    def query(self):
        res = cudart().cudaStreamQuery(self)
        if res == ERROR_NOT_READY:
            return False
        check_error(res)
        return True

    def synchronize(self):
        check_error(cudart().cudaStreamSynchronize(self))

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.cuda_stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return o.device == self.device and o.cuda_stream == self.cuda_stream
        return False

    def __hash__(self):
        return hash((self.cuda_stream, self.device))

    def __repr__(self):
        return ('<torch.cuda.Stream device={0} cuda_stream={1:#x}>'
                .format(self.device, self.cuda_stream))


class Event(object):
    DEFAULT = 0x0
    BLOCKING_SYNC = 0x1
    DISABLE_TIMING = 0x2
    INTERPROCESS = 0x4

    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        flags = Event.DEFAULT
        if not enable_timing:
            flags |= Event.DISABLE_TIMING
        if blocking:
            flags |= Event.BLOCKING_SYNC
        if interprocess:
            flags |= Event.INTERPROCESS

        ptr = ctypes.c_void_p()
        self._cudart = cudart()
        check_error(self._cudart.cudaEventCreateWithFlags(
            ctypes.byref(ptr), ctypes.c_uint(flags)))
        self._as_parameter_ = ptr

    def __del__(self):
        if hasattr(self, '_as_parameter_'):
            check_error(self._cudart.cudaEventDestroy(self._as_parameter_))
            del self._as_parameter_

    def record(self, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream()
        stream.record_event(self)

    def wait(self, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream()
        stream.wait_event(self)

    def query(self):
        res = cudart().cudaEventQuery(self)
        if res == ERROR_NOT_READY:
            return False
        check_error(res)
        return True

    def elapsed_time(self, end_event):
        time_ms = ctypes.c_float()
        check_error(cudart().cudaEventElapsedTime(
            ctypes.byref(time_ms), self, end_event))
        return time_ms.value

    def synchronize(self):
        check_error(cudart().cudaEventSynchronize(self))

    def __repr__(self):
        return '<torch.cuda.Event {0:#x}>'.format(self._as_parameter_.value)
