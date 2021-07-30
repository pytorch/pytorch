"""Hermetic wrapper for coordinating between worker and parent.

When we pass a string to a subprocess worker for execution, we need some
mechanism for the caller to monitor progress to know when execution is
complete, or if there was a failure. (And if so, what it was.)

To facilitate this communication, we define `_subprocess_snippet_handler`,
and then extract the source and re-define it in the worker.
"""

def _subprocess_snippet_handler(
    snippet: str,
    begin_fpath: str,
    success_fpath: str,
    failure_fpath: str,
    finished_fpath: str,
) -> None:
    """Note that this function MUST be hermetic."""
    import datetime
    import pathlib
    import sys
    import traceback

    def log_progress(suffix: str) -> None:
        now = datetime.datetime.now().strftime("[%Y-%m-%d] %H:%M:%S.%f")
        print(f"\n{now}: TIMER_SUBPROCESS_{suffix}")

    log_progress("BEGIN")
    pathlib.Path(begin_fpath).touch()

    try:
        exec(
            compile(snippet, "<subprocess-worker>", "exec"),
            globals(),
        )

        log_progress("SUCCESS")
        pathlib.Path(success_fpath).touch()

    except Exception:
        # TODO: better error handling.
        log_progress("FAILED")
        pathlib.Path(failure_fpath).touch()
        traceback.print_exc()

    finally:
        log_progress("FINISHED")
        pathlib.Path(finished_fpath).touch()

        sys.stdout.flush()
        sys.stderr.flush()


import inspect
_RAW_SOURCE = inspect.getsource(_subprocess_snippet_handler)

# `python -i` treats and empty line as a break in the input, so we cannot
# have any in our handler. This also means that _subprocess_snippet_handler
# MUST not contain any multi-line string literals with empty lines, as this
# transformation would not be correct.
HANDLER_SOURCE = "\n".join([
    l for l in _RAW_SOURCE.splitlines(keepends=False)
    if l.strip()
])
