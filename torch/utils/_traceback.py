from types import TracebackType
import tempfile
import contextlib
import inspect

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
                # Don't delete the temporary file so the user can inspect it
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".py") as f:
                    f.write(source)
                # Create a frame.  Python doesn't let you construct
                # FrameType directly, so just make one with compile
                frame = tb.tb_frame
                code = compile('__inspect_currentframe()', f.name, 'eval')
                # Python 3.8 only.  In earlier versions of Python
                # just have less accurate name info
                if hasattr(code, 'replace'):
                    code = code.replace(co_name=frame.f_code.co_name)
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
