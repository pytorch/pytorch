from torch._C import _get_cpp_backtrace

FUNC_FILTERS = ['_PyEval', 'PyEval_', '_PyObject', 'do_call_core', 'PyVectorcall_', 'PyRun_']

def get_cpp_backtrace(frames_to_skip=0, maximum_number_of_frames=64) -> str:
    r"""
    Return a string containing the C++ stack trace of the current thread.

    Args:
        frames_to_skip (int): the number of frames to skip from the top of the stack
        maximum_number_of_frames (int): the maximum number of frames to return
    """
    return _get_cpp_backtrace(frames_to_skip, maximum_number_of_frames)

def get_python_cpp_trace():
    # warning: this function is for debugging purpose and is slow in secs
    from torch._C._profiler import gather_traceback, symbolize_tracebacks
    tb = symbolize_tracebacks([gather_traceback(python=True, script=True, cpp=True)])[0]
    tb = [
        x for x in tb
        if not any(x['name'].startswith(filter) for filter in FUNC_FILTERS)
    ]
    frames_to_skip = next(
        ind + 1 for ind, x in enumerate(tb)
        if x['name'] == 'get_python_cpp_trace'
        and x['filename'].endswith('torch/utils/cpp_backtrace.py')
    )
    tb = tb[frames_to_skip:]
    tb = [f"#{ind} {x['name']} {x['filename']}:{x['line']}" for ind, x in enumerate(tb)]
    return '\n'.join(tb)
