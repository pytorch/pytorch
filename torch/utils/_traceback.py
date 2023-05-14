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
                    code = code.replace(
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
