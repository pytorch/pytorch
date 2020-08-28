import ctypes

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
        rangePushEx = _fail
        markEx = _fail
        version = None
        size = None


    _nvtx = _NVTXStub()  # type: ignore[assignment]

__all__ = ['range_push', 'range_pop', 'mark']

#TODO: Should these be pulled in from nvToolsExt.h directly?
colors = {
        'blue':        0x003498db,
        'green':       0x002ecc71,
        'yellow':      0x00f1c40f,
        'orange':      0x00e67e22,
        'red':         0x00e74c3c,
        'purple':      0x009b59b6,
        'navy':        0x0034495e,
        'gray':        0x0095a5a6,
        'silver':      0x00bdc3c7,
        'darkgray':    0x007f8c8d,
        }

def range_push(msg,color='silver'):
    """
    Pushes a range onto a stack of nested range span.  Returns zero-based
    depth of the range that is started.

    Arguments:
        msg (string): ASCII message to associate with range
    """
    attrib = EventAttributes(msg=msg,color=colors[color])
    return _nvtx.rangePushEx(attrib)


def range_pop():
    """
    Pops a range off of a stack of nested range spans.  Returns the
    zero-based depth of the range that is ended.
    """
    return _nvtx.rangePop()


def mark(msg,color='silver'):
    """
    Describe an instantaneous event that occurred at some point.

    Arguments:
        msg (string): ASCII message to associate with the event.
    """
    attrib = EventAttributes(msg=msg,color=colors[color])
    return _nvtx.markEx(attrib)


def EventAttributes(version=_nvtx.version, 
                    size=_nvtx.size, 
                    colorType=int(_nvtx.NVTX_COLOR_ARGB),
                    color=colors['yellow'],
                    msgType=int(_nvtx.NVTX_MESSAGE_TYPE_ASCII),
                    msg=''):
    """
    This uses instead the pybind included class, to ease the message usage
    """
    attrib = _nvtx.nvtxEventAttributes_t()
    attrib.version=version
    attrib.size=size
    attrib.colorType=colorType
    attrib.color=color
    attrib.messageType=msgType
    attrib.message=_nvtx.nvtxMessageValue_t() #use pybind11 instead of ctypes for this
    attrib.message.ascii=msg.encode("utf-8")
    return attrib

