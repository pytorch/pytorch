import glob
import json
import locale
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any


_ORIGINAL_POPEN = subprocess.Popen
_ORIGINAL_CALL = subprocess.call
_ORIGINAL_RUN = subprocess.run
_INSTALLED = False
_LOG_LOCK = threading.Lock()
_MAX_TEXT = 1000


def _debug_dir() -> str:
    path = os.path.join(os.path.dirname(__file__), "test-reports", "subprocess-debug")
    os.makedirs(path, exist_ok=True)
    return path


def _log_path() -> str:
    return os.path.join(_debug_dir(), f"subprocess_{os.getpid()}.jsonl")


def _output_path(stream: str) -> str:
    fd, path = tempfile.mkstemp(
        dir=_debug_dir(), prefix=f"subprocess_{os.getpid()}_", suffix=f".{stream}"
    )
    os.close(fd)
    return path


def _short_text(value: Any) -> str:
    if isinstance(value, bytes):
        text = os.fsdecode(value)
    else:
        text = str(value)
    if len(text) > _MAX_TEXT:
        return text[:_MAX_TEXT] + "...<truncated>"
    return text


def _command_text(args: Any) -> Any:
    if isinstance(args, (list, tuple)):
        return [_short_text(arg) for arg in args]
    return _short_text(args)


def _stdio_info(value: Any) -> Any:
    if value is None:
        return "inherit"
    if value == subprocess.PIPE:
        return "PIPE"
    if value == subprocess.STDOUT:
        return "STDOUT"
    if value == subprocess.DEVNULL:
        return "DEVNULL"
    name = getattr(value, "name", None)
    if name is not None:
        return {"file": _short_text(name)}
    return _short_text(value)


def _spawn_stack() -> list[str]:
    stack = traceback.extract_stack()[:-2]
    lines = []
    for frame in stack:
        filename = frame.filename
        if filename.endswith("subprocess.py") or filename.endswith("subprocess_debug.py"):
            continue
        try:
            filename = os.path.relpath(
                filename, os.path.dirname(os.path.dirname(__file__))
            )
        except ValueError:
            pass
        lines.append(f"{filename}:{frame.lineno} in {frame.name}")
    return lines[-8:]


def _write_record(record: dict[str, Any]) -> None:
    record["time"] = time.time()
    try:
        with _LOG_LOCK:
            with open(_log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        pass


class _TracedPopen(_ORIGINAL_POPEN):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        start_time = time.time()
        command = args[0] if args else kwargs.get("args")
        try:
            super().__init__(*args, **kwargs)
        except Exception as exc:
            _write_record(
                {
                    "event": "spawn_error",
                    "parent_pid": os.getpid(),
                    "command": _command_text(command),
                    "cwd": _short_text(kwargs.get("cwd") or os.getcwd()),
                    "shell": bool(kwargs.get("shell", False)),
                    "error": repr(exc),
                    "stack": _spawn_stack(),
                }
            )
            raise

        self._pytorch_subprocess_debug_start = start_time
        self._pytorch_subprocess_debug_finished = False
        self._pytorch_subprocess_debug_trace_id = (
            f"{os.getpid()}:{self.pid}:{start_time}"
        )
        _write_record(
            {
                "event": "start",
                "trace_id": self._pytorch_subprocess_debug_trace_id,
                "parent_pid": os.getpid(),
                "child_pid": self.pid,
                "command": _command_text(command),
                "cwd": _short_text(kwargs.get("cwd") or os.getcwd()),
                "shell": bool(kwargs.get("shell", False)),
                "stdout": _stdio_info(kwargs.get("stdout")),
                "stderr": _stdio_info(kwargs.get("stderr")),
                "stack": _spawn_stack(),
            }
        )

    def _record_finish(self) -> None:
        if getattr(self, "_pytorch_subprocess_debug_finished", False):
            return
        if self.returncode is None:
            return
        self._pytorch_subprocess_debug_finished = True
        start_time = getattr(self, "_pytorch_subprocess_debug_start", None)
        elapsed = time.time() - start_time if start_time is not None else None
        _write_record(
            {
                "event": "finish",
                "trace_id": getattr(self, "_pytorch_subprocess_debug_trace_id", None),
                "parent_pid": os.getpid(),
                "child_pid": self.pid,
                "returncode": self.returncode,
                "elapsed_seconds": elapsed,
            }
        )

    def wait(self, *args: Any, **kwargs: Any) -> int:
        try:
            return super().wait(*args, **kwargs)
        finally:
            self._record_finish()

    def communicate(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        try:
            return super().communicate(*args, **kwargs)
        finally:
            self._record_finish()

    def poll(self) -> int | None:
        result = super().poll()
        self._record_finish()
        return result


def _text_mode(kwargs: dict[str, Any]) -> bool:
    return bool(
        kwargs.get("text")
        or kwargs.get("universal_newlines")
        or kwargs.get("encoding")
        or kwargs.get("errors")
    )


def _read_output_file(path: str, kwargs: dict[str, Any]) -> bytes | str:
    with open(path, "rb") as f:
        data = f.read()
    if not _text_mode(kwargs):
        return data
    encoding = kwargs.get("encoding") or locale.getpreferredencoding(False)
    errors = kwargs.get("errors") or "strict"
    return data.decode(encoding, errors)


def _maybe_spool_outputs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], list[Any], dict[str, str]]:
    spooled_kwargs = kwargs.copy()
    files = []
    paths = {}

    if spooled_kwargs.get("stdout") == subprocess.PIPE:
        paths["stdout"] = _output_path("stdout")
        stdout_file = open(paths["stdout"], "w+b")
        files.append(stdout_file)
        spooled_kwargs["stdout"] = stdout_file

    if spooled_kwargs.get("stderr") == subprocess.PIPE:
        paths["stderr"] = _output_path("stderr")
        stderr_file = open(paths["stderr"], "w+b")
        files.append(stderr_file)
        spooled_kwargs["stderr"] = stderr_file

    return spooled_kwargs, files, paths


def _close_files(files: list[Any]) -> None:
    for f in files:
        try:
            f.close()
        except Exception:
            pass


def _traced_run(
    *popenargs: Any,
    input: Any = None,
    capture_output: bool = False,
    timeout: float | None = None,
    check: bool = False,
    **kwargs: Any,
) -> Any:
    if input is not None:
        if kwargs.get("stdin") is not None:
            raise ValueError("stdin and input arguments may not both be used.")
        kwargs["stdin"] = subprocess.PIPE

    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError(
                "stdout and stderr arguments may not be used with capture_output."
            )
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    captures_stdout = kwargs.get("stdout") == subprocess.PIPE
    captures_stderr = kwargs.get("stderr") == subprocess.PIPE
    if not captures_stdout and not captures_stderr:
        return _ORIGINAL_RUN(
            *popenargs,
            input=input,
            capture_output=False,
            timeout=timeout,
            check=check,
            **kwargs,
        )

    spooled_kwargs, files, paths = _maybe_spool_outputs(kwargs)
    try:
        result = _ORIGINAL_RUN(
            *popenargs,
            input=input,
            capture_output=False,
            timeout=timeout,
            check=False,
            **spooled_kwargs,
        )
        if captures_stdout:
            result.stdout = _read_output_file(paths["stdout"], kwargs)
        if captures_stderr:
            result.stderr = _read_output_file(paths["stderr"], kwargs)
        if check and result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )
        return result
    except subprocess.TimeoutExpired as exc:
        if captures_stdout:
            exc.output = _read_output_file(paths["stdout"], kwargs)
            exc.stdout = exc.output
        if captures_stderr:
            exc.stderr = _read_output_file(paths["stderr"], kwargs)
        raise
    finally:
        _close_files(files)


def _traced_call(
    *popenargs: Any, timeout: float | None = None, **kwargs: Any
) -> int:
    captures_stdout = kwargs.get("stdout") == subprocess.PIPE
    captures_stderr = kwargs.get("stderr") == subprocess.PIPE
    if not captures_stdout and not captures_stderr:
        return _ORIGINAL_CALL(*popenargs, timeout=timeout, **kwargs)

    spooled_kwargs, files, _ = _maybe_spool_outputs(kwargs)
    try:
        return _ORIGINAL_CALL(*popenargs, timeout=timeout, **spooled_kwargs)
    finally:
        _close_files(files)


def install() -> None:
    global _INSTALLED
    if _INSTALLED or subprocess.Popen is _TracedPopen:
        return
    subprocess.Popen = _TracedPopen
    subprocess.run = _traced_run
    subprocess.call = _traced_call
    _INSTALLED = True


def _read_file(path: str) -> str:
    try:
        with open(path, errors="replace") as f:
            return f.read().strip()
    except Exception as exc:
        return f"<unavailable: {exc!r}>"


def _read_cmdline(pid: int) -> str:
    text = _read_file(f"/proc/{pid}/cmdline")
    return text.replace("\x00", " ").strip()


def _read_link(path: str) -> str:
    try:
        return os.readlink(path)
    except Exception as exc:
        return f"<unavailable: {exc!r}>"


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _children(pid: int) -> list[int]:
    children: list[int] = []
    for path in glob.glob(f"/proc/{pid}/task/*/children"):
        for child in _read_file(path).split():
            try:
                children.append(int(child))
            except ValueError:
                pass
    return sorted(set(children))


def _descendants(pid: int) -> list[int]:
    seen: set[int] = set()
    pending = [pid]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(_children(current))
    return [p for p in seen if p != pid]


def _dump_proc_state(pid: int, out: Any) -> None:
    print(f"--- /proc state for pid={pid} ---", file=out)
    print(f"cmdline: {_read_cmdline(pid)}", file=out)
    print(f"cwd: {_read_link(f'/proc/{pid}/cwd')}", file=out)
    print(f"wchan: {_read_file(f'/proc/{pid}/wchan')}", file=out)
    print(f"syscall: {_read_file(f'/proc/{pid}/syscall')}", file=out)
    print(_read_file(f"/proc/{pid}/status"), file=out)


def _dump_output_file(label: str, path: str, out: Any) -> None:
    try:
        with open(path, "rb") as f:
            content = f.read().decode("utf-8", "replace")
    except Exception as exc:
        content = f"<unavailable: {exc!r}>"
    print(f"--- captured {label} path={path} ---", file=out)
    print(content, file=out)


def _dump_record_output(record: dict[str, Any], out: Any) -> None:
    seen: set[str] = set()
    for label in ("stdout", "stderr"):
        info = record.get(label)
        if not isinstance(info, dict):
            continue
        path = info.get("file")
        if not isinstance(path, str) or path in seen:
            continue
        seen.add(path)
        _dump_output_file(label, path, out)


def _recent_records(max_age_seconds: float) -> list[dict[str, Any]]:
    cutoff = time.time() - max_age_seconds
    records: list[dict[str, Any]] = []
    try:
        paths = glob.glob(os.path.join(_debug_dir(), "subprocess_*.jsonl"))
    except Exception:
        return records
    for path in paths:
        try:
            if os.path.getmtime(path) < cutoff:
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("time", 0.0) >= cutoff:
                        records.append(record)
        except Exception:
            pass
    return records


def dump_recent_subprocess_traces(
    reason: str, max_age_seconds: float = 1800.0
) -> None:
    records = _recent_records(max_age_seconds)
    starts: dict[str, dict[str, Any]] = {}
    finished: set[str] = set()
    for record in records:
        trace_id = record.get("trace_id")
        child_pid = record.get("child_pid")
        if not isinstance(trace_id, str):
            continue
        if not isinstance(child_pid, int):
            continue
        if record.get("event") == "start":
            starts[trace_id] = record
        elif record.get("event") == "finish":
            finished.add(trace_id)

    active = [
        record
        for trace_id, record in starts.items()
        if trace_id not in finished and _process_exists(record["child_pid"])
    ]
    if not active:
        return

    out = sys.__stderr__ or sys.stderr
    print(
        f"\n===== PYTORCH SUBPROCESS DEBUG BEGIN reason={reason} =====",
        file=out,
        flush=True,
    )
    for record in sorted(active, key=lambda record: record["child_pid"]):
        pid = record["child_pid"]
        print(json.dumps(record, sort_keys=True), file=out)
        _dump_record_output(record, out)
        _dump_proc_state(pid, out)
        for child in _descendants(pid):
            _dump_proc_state(child, out)
    print("===== PYTORCH SUBPROCESS DEBUG END =====\n", file=out, flush=True)
