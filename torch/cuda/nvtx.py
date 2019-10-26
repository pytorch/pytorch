try:
    from torch._C import _nvtx
except ImportError:
    class _NVTXStub(object):
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError("NVTX functions not installed. Are you sure you have a CUDA build?")

        rangePushA = _fail
        rangePop = _fail
        markA = _fail

    _nvtx = _NVTXStub()

__all__ = ['range_push', 'range_pop', 'mark']


def range_push(msg):
    """
    Pushes a range onto a stack of nested range span.  Returns zero-based
    depth of the range that is started.

    Arguments:
        msg (string): ASCII message to associate with range
    """
    return _nvtx.rangePushA(msg)


def range_pop():
    """
    Pops a range off of a stack of nested range spans.  Returns the
    zero-based depth of the range that is ended.
    """
    return _nvtx.rangePop()


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Arguments:
        msg (string): ASCII message to associate with the event.
    """
    return _nvtx.markA(msg)
