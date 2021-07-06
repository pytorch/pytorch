try:
    from torch._C import _itt
except ImportError:
    class _ITTStub(object):
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError("ITT functions not installed. Are you sure you have a ITT build?")

        rangePush = _fail
        rangePop = _fail
        mark = _fail

    _itt = _ITTStub()


__all__ = ['range_push', 'range_pop', 'mark']


def range_push(msg):
    """
    Arguments:
        msg (string): ASCII message to associate with range
    """
    return _itt.rangePush(msg)


def range_pop():
    """
    """
    return _itt.rangePop()


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.
    Arguments:
        msg (string): ASCII message to associate with the event.
    """
    return _itt.mark(msg)
