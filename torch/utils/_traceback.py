from types import TracebackType
import tempfile
import traceback
import contextlib
import inspect
import os.path

# This file contains utilities for ensuring dynamically compile()'d
# code fragments display their line numbers in backtraces.
#
# The constraints:
#
# - We don't have control over the user exception printer (in particular,
#   we cannot assume the linecache trick will work, c.f.
#   https://stackoverflow.com/q/50515651/23845 )
#
# - We don't want to create temporary files every time we compile()
#   some code; file creation should happen lazily only at exception
#   time.  Arguably, you *should* be willing to write out your
#   generated Python code to file system, but in some situations
#   (esp. library code) it would violate user expectation to write
#   to the file system, so we try to avoid it.  In particular, we'd
#   like to keep the files around, so users can open up the files
#   mentioned in the trace; if the file is invisible, we want to
#   avoid clogging up the filesystem.
#
#   If this is not a constraint for you, there is a substantially simpler
#   way to implement the functionality in this PR: instead of using
#   eval/exec directly, just always write a Python file to filesystem
#   and compile that.
#
# - You have control over a context where the compiled code will get
#   executed, so that we can interpose while the stack is unwinding
#   (otherwise, we have no way to interpose on the exception printing
#   process.)
#
# There are two things you have to do to make use of the utilities here:
#
# - When you compile your source code, you must save its string source
#   in its f_globals under the magic name "__compile_source__"
#
# - Before running the compiled code, enter the
#   report_compile_source_on_error() context manager.

@contextlib.contextmanager
def report_compile_source_on_error():
    try:
        yield
    except Exception as exc:
        tb = exc.__traceback__

        # Walk the traceback, looking for frames that have
        # source attached
        stack = []
        while tb is not None:
            filename = tb.tb_frame.f_code.co_filename
            source = tb.tb_frame.f_globals.get("__compile_source__")

            if filename == "<string>" and source is not None:
                # What black magic are we doing here?  Intuitively, what
                # we would like to do is overwrite the co_filename on any
                # frames that were generated from exec/eval so that they
                # point to a temporary file that has the actual line
                # information, so Python's default error printer can print
                # useful line information on it.
                #
                # Writing out the temporary file is easy.  But overwriting
                # co_filename is not!  You can't modify the code object
                # associated with a frame.  You can, however, reconstruct
                # a traceback with entirely new frames from scratch, so that's
                # what we do.  But there's another problem, which is how to
                # make the frame?
                #
                # The black magic is we make a frankenstein frame and code
                # object which resembles the original frame/code enough so
                # that it will print properly under traceback and the default
                # error printer, but IT IS NOT THE ORIGINAL FRAME (you
                # couldn't, e.g., execute its code with different variables
                # and expect it to work.)

                # Don't delete the temporary file so the user can inspect it
                # TODO: This creates a temporary file for every frame, but we
                # technically only need one per distinct __compile_source__
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".py") as f:
                    f.write(source)
                # Create a frame.  Python doesn't let you construct
                # FrameType directly, so just make one with compile
                frame = tb.tb_frame
                code = compile('__inspect_currentframe()', f.name, 'eval')
                code = code.replace(co_name=frame.f_code.co_name)
                # Python 3.11 only
                if hasattr(frame.f_code, 'co_linetable'):
                    # We can't copy ALL of the metadata over, because you
                    # can cause Python to segfault this way.  What exactly
                    # do we need?  We need enough information for
                    # traceback to be able to print the exception
                    # correctly.  Code reading Lib/traceback.py reveals
                    # that traceback calls code.co_positions() in order to
                    # get the augmented line/col numbers.  Objects/codeobject.c,
                    # specifically _PyCode_InitAddressRange, reveals that
                    # this iterator is initialized from co_linetable and
                    # co_firstfileno.  So copy these we must!
                    code = code.replace(  # type: ignore[call-arg]
                        co_linetable=frame.f_code.co_linetable,  # type: ignore[attr-defined]
                        co_firstlineno=frame.f_code.co_firstlineno,  # type: ignore[attr-defined]
                    )
                fake_frame = eval(
                    code,
                    frame.f_globals,
                    {
                        **frame.f_locals,
                        '__inspect_currentframe': inspect.currentframe
                    }
                )
                fake_tb = TracebackType(
                    None, fake_frame, tb.tb_lasti, tb.tb_lineno
                )
                stack.append(fake_tb)
            else:
                stack.append(tb)

            tb = tb.tb_next

        # Reconstruct the linked list
        tb_next = None
        for tb in reversed(stack):
            tb.tb_next = tb_next
            tb_next = tb

        raise exc.with_traceback(tb_next)

def shorten_filename(fn):
    """
    Shorten a source filepath, under the assumption that anything under torch/
    directory is "obvious" and doesn't need to be shown to user.
    """
    # Truncate torch/foo.py to foo.py
    prefix = os.path.commonpath([fn, os.path.join(os.path.dirname(os.path.dirname(__file__)), "")])
    return fn[len(prefix)+1:]

def format_frame(frame):
    """
    Format a FrameSummary in a short way, without printing full absolute path
    or code.  The idea is the result fits on a single line.
    """
    return f"{shorten_filename(frame.filename)}:{frame.lineno} in {frame.name}"

def format_traceback_short(tb):
    """
    Format a TracebackType in a short way, printing only the inner-most frame.
    """
    return format_frame(traceback.extract_tb(tb)[-1])

class CapturedTraceback:
    __slots__ = ['tb', 'skip', '_summary']

    def __init__(self, tb, skip=0):
        self.tb = tb
        self.skip = 0
        # Cached StackSummary; this mostly exists so CapturedTraceback
        # can be reliably pickled and then printed later
        self._summary = None

    def summary(self):
        import torch._C._profiler

        if self._summary is None:
            assert self.tb is not None
            self._summary = _extract_symbolized_tb(
                torch._C._profiler.symbolize_tracebacks([self.tb])[0],
                self.skip
            )
        return self._summary

    def __getstate__(self):
        # Force populate summary
        self.summary()
        return (None, {
            'tb': None,  # TB is not pickleable
            'skip': self.skip,
            '_summary': self._summary
        })

    @staticmethod
    def extract(*, script=False, cpp=False, skip=0):
        """
        Like traceback.extract_stack(), but faster (approximately 20x faster); it
        is fast enough that you can unconditionally log stacks this way as part of
        normal execution.  It returns a torch._C._profiler.CapturedTraceback
        object that must be formatted specially with format_captured_tb.

        By default, this only reports Python backtraces (like extract_stack).  You
        can set the script/cpp kwargs to also turn on TorchScript/C++ trace
        reporting.
        """
        import torch._C._profiler

        if script or cpp:
            assert skip == 0, "skip with script/cpp NYI"

        return CapturedTraceback(
            torch._C._profiler.gather_traceback(python=True, script=script, cpp=cpp),
            # Elide extract() frame if we don't have script/cpp frames.  If
            # we do have those frames, it doesn't work so force zero.
            0 if script or cpp else skip + 1
        )

    def format(self):
        """
        Formats a single torch._C._profiler.CapturedTraceback into a list of
        strings equivalent to the output of traceback.format_list.  Note that if
        pass it CapturedTraceback with C++ traces,  it is better not to use this
        function and use the batch formatting API format_captured_tbs to amortize
        the cost of symbolization
        """
        return traceback.format_list(self.summary())

    @staticmethod
    def format_all(tbs):
        """
        Bulk version of CapturedTraceback.format.  Returns a list of list of strings.
        """
        import torch._C._profiler

        # Directly populate tracebacks that already have cached summaries
        rs = []
        delayed_idxs = []
        for i, tb in enumerate(tbs):
            if tb._summary is not None:
                rs.append(traceback.format_list(tb._summary))
            else:
                rs.append(None)
                delayed_idxs.append(i)

        stbs = torch._C._profiler.symbolize_tracebacks([tbs[i].tb for i in delayed_idxs])
        for i, stb in zip(delayed_idxs, stbs):
            tb = tbs[i]
            tb._summary = _extract_symbolized_tb(stb, tb.skip)
            rs[i] = traceback.format_list(tb._summary)

        return rs


def _extract_symbolized_tb(tb, skip):
    """
    Given a symbolized traceback from symbolize_tracebacks, return a StackSummary object of
    pre-processed stack trace entries.
    """
    stack = traceback.StackSummary()
    for f in reversed(tb[skip:]):
        stack.append(traceback.FrameSummary(f['filename'], f['line'], f['name']))
    return stack
